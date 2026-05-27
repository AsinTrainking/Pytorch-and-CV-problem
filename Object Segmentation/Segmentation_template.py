# mask_rcnn_segmentation_template_bilingual.py  # EN: Mask R-CNN instance segmentation template based on your detection code. CN: 基于你当前 detection 代码修改得到的 Mask R-CNN 实例分割模板。

import os  # EN: Import os for file and folder path operations. CN: 导入 os，用于文件和文件夹路径操作。
import time  # EN: Import time for measuring training time. CN: 导入 time，用于统计训练时间。
import zipfile  # EN: Import zipfile for extracting downloaded dataset. CN: 导入 zipfile，用于解压下载的数据集。
import random  # EN: Import random for reproducibility. CN: 导入 random，用于设置随机种子。
import urllib.request  # EN: Import urllib for downloading dataset. CN: 导入 urllib，用于下载数据集。

import torch  # EN: Import PyTorch core library. CN: 导入 PyTorch 核心库。
import torchvision  # EN: Import torchvision for vision models and utilities. CN: 导入 torchvision，用于视觉模型和工具函数。
import torch.utils.data  # EN: Import PyTorch dataset and dataloader utilities. CN: 导入 PyTorch 数据集和 DataLoader 工具。

from PIL import Image  # EN: Import PIL for reading images. CN: 导入 PIL，用于读取图像。
import numpy as np  # EN: Import NumPy for mask processing. CN: 导入 NumPy，用于 mask 处理。
import matplotlib.pyplot as plt  # EN: Import matplotlib for visualization. CN: 导入 matplotlib，用于结果可视化。

from torchvision.models.detection import maskrcnn_resnet50_fpn  # EN: Import pretrained Mask R-CNN model. CN: 导入预训练 Mask R-CNN 模型。
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # EN: Import predictor for replacing box classification head. CN: 导入 FastRCNNPredictor，用于替换 box 分类头。
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor  # EN: Import predictor for replacing mask prediction head. CN: 导入 MaskRCNNPredictor，用于替换 mask 分割头。
from torchvision.transforms import v2 as T  # EN: Import torchvision v2 transforms for image-target transforms. CN: 导入 torchvision v2 transforms，用于同时处理图像和 target。
from torchvision.utils import draw_bounding_boxes  # EN: Import function to draw bounding boxes. CN: 导入画 bounding box 的函数。
from torchvision.utils import draw_segmentation_masks  # EN: Import function to draw segmentation masks. CN: 导入画 segmentation mask 的函数。


# ============================================================
# 1. Basic configuration
# 1. 基础配置
# ============================================================

DATA_ROOT = "data"  # EN: Root folder for storing dataset. CN: 保存数据集的根目录。
DATASET_NAME = "PennFudanPed"  # EN: Dataset folder name. CN: 数据集文件夹名称。
DATASET_ZIP_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"  # EN: URL for downloading PennFudan dataset. CN: PennFudan 数据集下载链接。
OUTPUT_DIR = "outputs_segmentation"  # EN: Output folder for checkpoints and plots. CN: 保存 checkpoint 和曲线图的输出文件夹。
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "maskrcnn_resnet50_fpn_pennfudan.pth")  # EN: Path for saving trained Mask R-CNN checkpoint. CN: 保存训练后 Mask R-CNN checkpoint 的路径。

CLASS_NAMES = ["background", "pedestrian"]  # EN: Class 0 is background, class 1 is pedestrian. CN: 类别 0 是背景，类别 1 是行人。
NUM_CLASSES = len(CLASS_NAMES)  # EN: Number of classes including background. CN: 类别数量，包括背景类。
BATCH_SIZE = 2  # EN: Batch size for training. CN: 训练 batch size。
NUM_EPOCHS = 5  # EN: Number of training epochs. CN: 训练 epoch 数量。
LEARNING_RATE = 0.005  # EN: Learning rate for SGD optimizer. CN: SGD 优化器学习率。
MOMENTUM = 0.9  # EN: Momentum for SGD optimizer. CN: SGD 优化器动量。
WEIGHT_DECAY = 0.0005  # EN: Weight decay for regularization. CN: 权重衰减，用于正则化。
LR_STEP_SIZE = 3  # EN: Learning rate decay step size. CN: 学习率衰减间隔。
LR_GAMMA = 0.1  # EN: Learning rate decay factor. CN: 学习率衰减系数。
NUM_WORKERS = 0  # EN: Use 0 for Windows/Jupyter safety. CN: Windows/Jupyter 中建议使用 0。
TRAIN_RATIO = 0.8  # EN: Ratio of images used for training. CN: 用于训练的数据比例。
SEED = 42  # EN: Random seed for reproducibility. CN: 随机种子，保证结果尽量可复现。
SCORE_THRESHOLD = 0.5  # EN: Confidence threshold for visualizing predictions. CN: 可视化预测结果时的置信度阈值。
MASK_THRESHOLD = 0.5  # EN: Probability threshold for converting predicted masks to binary masks. CN: 将预测 mask 概率图转为二值 mask 的阈值。


# ============================================================
# 2. Reproducibility and device
# 2. 可复现性和设备选择
# ============================================================

def set_seed(seed=42):  # EN: Define a function to set random seeds. CN: 定义设置随机种子的函数。
    random.seed(seed)  # EN: Set Python random seed. CN: 设置 Python random 随机种子。
    np.random.seed(seed)  # EN: Set NumPy random seed. CN: 设置 NumPy 随机种子。
    torch.manual_seed(seed)  # EN: Set PyTorch CPU random seed. CN: 设置 PyTorch CPU 随机种子。
    torch.cuda.manual_seed_all(seed)  # EN: Set CUDA random seed if GPU is available. CN: 如果有 GPU，设置 CUDA 随机种子。


def get_device():  # EN: Define a function to select CPU or GPU automatically. CN: 定义自动选择 CPU 或 GPU 的函数。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # EN: Use GPU if available; otherwise CPU. CN: 如果有 GPU 就用 GPU，否则用 CPU。
    print(f"Using device: {device}")  # EN: Print selected device. CN: 打印当前使用的设备。
    if device.type == "cuda":  # EN: Check whether selected device is CUDA. CN: 检查当前设备是否是 CUDA。
        print(f"GPU: {torch.cuda.get_device_name(0)}")  # EN: Print GPU name. CN: 打印 GPU 名称。
    return device  # EN: Return selected device. CN: 返回当前设备。


# ============================================================
# 3. Dataset download and transforms
# 3. 数据集下载和预处理
# ============================================================

def download_and_extract_dataset():  # EN: Define a function to download and extract PennFudan dataset. CN: 定义下载并解压 PennFudan 数据集的函数。
    os.makedirs(DATA_ROOT, exist_ok=True)  # EN: Create data root folder if it does not exist. CN: 如果数据根目录不存在，则创建。
    dataset_dir = os.path.join(DATA_ROOT, DATASET_NAME)  # EN: Build full dataset directory path. CN: 构建完整数据集目录路径。
    zip_path = os.path.join(DATA_ROOT, "PennFudanPed.zip")  # EN: Build downloaded zip file path. CN: 构建下载后的 zip 文件路径。
    if os.path.exists(dataset_dir):  # EN: If dataset folder already exists. CN: 如果数据集文件夹已经存在。
        print(f"Dataset already exists: {dataset_dir}")  # EN: Print existing dataset path. CN: 打印已有数据集路径。
        return dataset_dir  # EN: Return existing dataset directory. CN: 返回已有数据集目录。
    print("Downloading dataset...")  # EN: Print download message. CN: 打印下载提示信息。
    print(f"URL: {DATASET_ZIP_URL}")  # EN: Print dataset URL. CN: 打印数据集下载链接。
    urllib.request.urlretrieve(DATASET_ZIP_URL, zip_path)  # EN: Download dataset zip file. CN: 下载数据集 zip 文件。
    print("Extracting dataset...")  # EN: Print extraction message. CN: 打印解压提示信息。
    with zipfile.ZipFile(zip_path, "r") as zip_ref:  # EN: Open zip file for reading. CN: 以只读方式打开 zip 文件。
        zip_ref.extractall(DATA_ROOT)  # EN: Extract all files into DATA_ROOT. CN: 将所有文件解压到 DATA_ROOT。
    print(f"Dataset extracted to: {dataset_dir}")  # EN: Print extracted dataset path. CN: 打印解压后的数据集路径。
    return dataset_dir  # EN: Return dataset directory. CN: 返回数据集目录。


def get_transform(train):  # EN: Define transforms for training or validation. CN: 定义训练或验证阶段的预处理。
    transform = []  # EN: Create an empty transform list. CN: 创建空的 transform 列表。
    if train:  # EN: If this is training transform. CN: 如果是训练阶段的 transform。
        transform.append(T.RandomHorizontalFlip(p=0.5))  # EN: Randomly flip both image and target masks/boxes. CN: 随机水平翻转图像以及 target 中的 mask/box。
    transform.append(T.ToImage())  # EN: Convert PIL image to torchvision image tensor. CN: 将 PIL 图像转换为 torchvision image tensor。
    transform.append(T.ToDtype(torch.float32, scale=True))  # EN: Convert image to float32 and scale pixels to [0,1]. CN: 将图像转为 float32，并缩放到 [0,1]。
    return T.Compose(transform)  # EN: Compose and return transforms. CN: 组合并返回 transforms。


# ============================================================
# 4. PennFudan instance segmentation dataset
# 4. PennFudan 实例分割数据集
# ============================================================

class PennFudanSegmentationDataset(torch.utils.data.Dataset):  # EN: Define custom dataset for instance segmentation. CN: 定义实例分割任务的自定义数据集。
    def __init__(self, root, transforms=None):  # EN: Initialize dataset. CN: 初始化数据集。
        self.root = root  # EN: Store dataset root path. CN: 保存数据集根目录。
        self.transforms = transforms  # EN: Store transforms. CN: 保存预处理方法。
        self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))  # EN: List all image filenames. CN: 列出所有图像文件名。
        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))  # EN: List all mask filenames. CN: 列出所有 mask 文件名。

    def __getitem__(self, idx):  # EN: Get one image and its target by index. CN: 根据索引读取一张图像及其 target。
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])  # EN: Build image path. CN: 构建图像路径。
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])  # EN: Build mask path. CN: 构建 mask 路径。
        img = Image.open(img_path).convert("RGB")  # EN: Read image and convert to RGB. CN: 读取图像并转换为 RGB。
        mask = Image.open(mask_path)  # EN: Read instance mask image. CN: 读取实例 mask 图像。
        mask = np.array(mask)  # EN: Convert mask to NumPy array. CN: 将 mask 转换为 NumPy 数组。
        obj_ids = np.unique(mask)  # EN: Get all unique object ids in the mask. CN: 获取 mask 中所有唯一的目标 ID。
        obj_ids = obj_ids[1:]  # EN: Remove background id 0. CN: 移除背景 ID 0。
        masks = mask == obj_ids[:, None, None]  # EN: Convert instance ids to binary masks [N,H,W]. CN: 将实例 ID 转换为二值 mask，形状为 [N,H,W]。
        boxes = []  # EN: Create an empty list for bounding boxes. CN: 创建空列表保存 bounding boxes。
        valid_masks = []  # EN: Create an empty list for valid masks. CN: 创建空列表保存有效 masks。
        for i in range(len(obj_ids)):  # EN: Loop over each object instance. CN: 遍历每个目标实例。
            pos = np.where(masks[i])  # EN: Get pixel coordinates of current mask. CN: 获取当前 mask 的像素坐标。
            if len(pos[0]) == 0 or len(pos[1]) == 0:  # EN: Skip empty masks. CN: 跳过空 mask。
                continue  # EN: Continue to next instance. CN: 继续处理下一个实例。
            xmin = np.min(pos[1])  # EN: Minimum x coordinate. CN: x 方向最小坐标。
            xmax = np.max(pos[1])  # EN: Maximum x coordinate. CN: x 方向最大坐标。
            ymin = np.min(pos[0])  # EN: Minimum y coordinate. CN: y 方向最小坐标。
            ymax = np.max(pos[0])  # EN: Maximum y coordinate. CN: y 方向最大坐标。
            if xmax <= xmin or ymax <= ymin:  # EN: Skip invalid boxes. CN: 跳过无效的 bounding box。
                continue  # EN: Continue to next instance. CN: 继续处理下一个实例。
            boxes.append([xmin, ymin, xmax, ymax])  # EN: Save bounding box in [xmin,ymin,xmax,ymax] format. CN: 以 [xmin,ymin,xmax,ymax] 格式保存 box。
            valid_masks.append(masks[i])  # EN: Save corresponding valid mask. CN: 保存对应的有效 mask。
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # EN: Convert boxes to float tensor. CN: 将 boxes 转换为 float tensor。
        masks = torch.as_tensor(np.array(valid_masks), dtype=torch.uint8)  # EN: Convert masks to uint8 tensor [N,H,W]. CN: 将 masks 转换为 uint8 tensor，形状 [N,H,W]。
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # EN: All objects are pedestrians with label 1. CN: 所有目标都是 pedestrian，标签为 1。
        image_id = torch.tensor([idx])  # EN: Store image id. CN: 保存图像 ID。
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # EN: Compute object area from boxes. CN: 根据 boxes 计算目标面积。
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)  # EN: PennFudan has no crowd annotations. CN: PennFudan 没有 crowd 标注。
        target = {  # EN: Build target dictionary required by Mask R-CNN. CN: 构建 Mask R-CNN 需要的 target 字典。
            "boxes": boxes,  # EN: Bounding boxes tensor [N,4]. CN: bounding boxes tensor，形状 [N,4]。
            "labels": labels,  # EN: Class labels tensor [N]. CN: 类别标签 tensor，形状 [N]。
            "masks": masks,  # EN: Instance masks tensor [N,H,W]. CN: 实例分割 masks tensor，形状 [N,H,W]。
            "image_id": image_id,  # EN: Image id tensor. CN: 图像 ID tensor。
            "area": area,  # EN: Object area tensor [N]. CN: 目标面积 tensor，形状 [N]。
            "iscrowd": iscrowd,  # EN: Crowd flag tensor [N]. CN: crowd 标志 tensor，形状 [N]。
        }  # EN: End target dictionary. CN: 结束 target 字典。
        if self.transforms is not None:  # EN: If transforms are provided. CN: 如果提供了 transforms。
            img, target = self.transforms(img, target)  # EN: Apply transforms to both image and target. CN: 同时对图像和 target 应用 transforms。
        return img, target  # EN: Return image and target. CN: 返回图像和 target。

    def __len__(self):  # EN: Return dataset length. CN: 返回数据集长度。
        return len(self.imgs)  # EN: Number of images. CN: 图像数量。


# ============================================================
# 5. Dataloader collate function
# 5. DataLoader 的 collate 函数
# ============================================================

def collate_fn(batch):  # EN: Define custom collate function for variable number of objects. CN: 为不同数量目标定义自定义 collate 函数。
    return tuple(zip(*batch))  # EN: Return tuple of image list and target list. CN: 返回 image list 和 target list。


# ============================================================
# 6. Build Mask R-CNN model
# 6. 构建 Mask R-CNN 模型
# ============================================================

def build_model(num_classes):  # EN: Build a Mask R-CNN model for instance segmentation. CN: 构建用于实例分割的 Mask R-CNN 模型。
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")  # EN: Load pretrained Mask R-CNN with ResNet50-FPN backbone. CN: 加载带 ResNet50-FPN backbone 的预训练 Mask R-CNN。
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features  # EN: Get input feature dimension of box classifier. CN: 获取 box 分类器输入特征维度。
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)  # EN: Replace box predictor for our class number. CN: 根据当前类别数替换 box predictor。
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # EN: Get input channels of mask predictor. CN: 获取 mask predictor 的输入通道数。
    hidden_layer = 256  # EN: Hidden layer size for mask predictor. CN: mask predictor 的隐藏层通道数。
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)  # EN: Replace mask predictor for our class number. CN: 根据当前类别数替换 mask predictor。
    return model  # EN: Return modified Mask R-CNN model. CN: 返回修改后的 Mask R-CNN 模型。


# ============================================================
# 7. Training
# 7. 训练函数
# ============================================================

def train_one_epoch(model, optimizer, dataloader, device, epoch):  # EN: Train Mask R-CNN for one epoch. CN: 训练 Mask R-CNN 一个 epoch。
    model.train()  # EN: Set model to training mode. CN: 将模型设置为训练模式。
    total_loss = 0.0  # EN: Accumulate total loss. CN: 累计总损失。
    start_time = time.time()  # EN: Record epoch start time. CN: 记录当前 epoch 开始时间。
    for batch_idx, (images, targets) in enumerate(dataloader):  # EN: Iterate over mini-batches. CN: 遍历每个 mini-batch。
        images = [img.to(device) for img in images]  # EN: Move every image tensor to device. CN: 将每张图像 tensor 移动到设备。
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # EN: Move every target tensor to device. CN: 将每个 target 中的 tensor 移动到设备。
        loss_dict = model(images, targets)  # EN: Forward pass returns detection and mask losses in training mode. CN: 训练模式下前向传播会返回 detection 和 mask 损失。
        losses = sum(loss for loss in loss_dict.values())  # EN: Sum all loss terms. CN: 将所有损失项相加。
        optimizer.zero_grad()  # EN: Clear old gradients. CN: 清空上一轮梯度。
        losses.backward()  # EN: Backpropagate gradients. CN: 反向传播计算梯度。
        optimizer.step()  # EN: Update model parameters. CN: 更新模型参数。
        total_loss += losses.item()  # EN: Add current batch loss to total loss. CN: 将当前 batch loss 累加到总 loss。
        if batch_idx % 10 == 0:  # EN: Print logs every 10 batches. CN: 每 10 个 batch 打印一次日志。
            loss_items = {k: round(v.item(), 4) for k, v in loss_dict.items()}  # EN: Convert loss dictionary to readable numbers. CN: 将 loss 字典转换为可读数值。
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] Loss: {losses.item():.4f} {loss_items}")  # EN: Print training status. CN: 打印训练状态。
    avg_loss = total_loss / len(dataloader)  # EN: Compute average loss of this epoch. CN: 计算当前 epoch 的平均 loss。
    elapsed = time.time() - start_time  # EN: Compute elapsed time. CN: 计算当前 epoch 耗时。
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")  # EN: Print average loss. CN: 打印平均 loss。
    print(f"Epoch [{epoch}] Time: {elapsed:.1f}s")  # EN: Print elapsed time. CN: 打印耗时。
    return avg_loss  # EN: Return average loss. CN: 返回平均 loss。


# ============================================================
# 8. Inference and visualization
# 8. 推理和可视化
# ============================================================

def run_inference(model, dataloader, device, score_threshold=0.5, mask_threshold=0.5, max_images=3):  # EN: Run inference and visualize boxes plus masks. CN: 执行推理并可视化 boxes 和 masks。
    model.eval()  # EN: Set model to evaluation mode. CN: 将模型设置为评估模式。
    images_shown = 0  # EN: Count displayed images. CN: 统计已显示图像数量。
    with torch.no_grad():  # EN: Disable gradients during inference. CN: 推理时不计算梯度。
        for images, targets in dataloader:  # EN: Iterate over test dataloader. CN: 遍历测试集 DataLoader。
            images_gpu = [img.to(device) for img in images]  # EN: Move images to device. CN: 将图像移动到设备。
            outputs = model(images_gpu)  # EN: Get model predictions. CN: 获取模型预测结果。
            for img, output, target in zip(images, outputs, targets):  # EN: Loop over each image prediction. CN: 遍历每张图像的预测结果。
                img_uint8 = (img * 255).to(torch.uint8).cpu()  # EN: Convert image from float [0,1] to uint8 [0,255]. CN: 将图像从 float [0,1] 转换为 uint8 [0,255]。
                scores = output["scores"].cpu()  # EN: Get prediction confidence scores. CN: 获取预测置信度。
                keep = scores >= score_threshold  # EN: Keep predictions above score threshold. CN: 保留置信度高于阈值的预测。
                boxes = output["boxes"].cpu()[keep]  # EN: Filter predicted boxes. CN: 筛选预测 boxes。
                labels = output["labels"].cpu()[keep]  # EN: Filter predicted labels. CN: 筛选预测类别。
                scores = scores[keep]  # EN: Filter scores. CN: 筛选置信度。
                masks = output["masks"].cpu()[keep]  # EN: Filter predicted masks with shape [N,1,H,W]. CN: 筛选预测 masks，形状 [N,1,H,W]。
                if masks.numel() > 0:  # EN: If at least one mask is predicted. CN: 如果至少预测到一个 mask。
                    binary_masks = masks[:, 0] >= mask_threshold  # EN: Convert mask probabilities to binary masks [N,H,W]. CN: 将 mask 概率图转换为二值 mask，形状 [N,H,W]。
                    drawn = draw_segmentation_masks(img_uint8, binary_masks, alpha=0.5)  # EN: Overlay masks on image. CN: 将 masks 叠加到图像上。
                else:  # EN: If no mask is predicted. CN: 如果没有预测到 mask。
                    drawn = img_uint8  # EN: Use original image. CN: 使用原始图像。
                label_texts = [f"{CLASS_NAMES[label.item()]} {score:.2f}" for label, score in zip(labels, scores)]  # EN: Build label strings. CN: 构建类别和置信度文本。
                if boxes.numel() > 0:  # EN: If there are predicted boxes. CN: 如果存在预测 boxes。
                    drawn = draw_bounding_boxes(drawn, boxes=boxes, labels=label_texts, width=3)  # EN: Draw bounding boxes and labels. CN: 绘制 bounding boxes 和标签。
                plt.figure(figsize=(8, 6))  # EN: Create figure. CN: 创建画布。
                plt.imshow(drawn.permute(1, 2, 0))  # EN: Convert CHW to HWC and display image. CN: 将 CHW 转为 HWC 并显示图像。
                plt.axis("off")  # EN: Hide axes. CN: 隐藏坐标轴。
                plt.title("Predicted Instance Segmentation Masks")  # EN: Set title. CN: 设置标题。
                plt.show()  # EN: Show figure. CN: 显示图像。
                images_shown += 1  # EN: Increase displayed image count. CN: 已显示图像数量加一。
                if images_shown >= max_images:  # EN: Stop after max_images. CN: 达到最大显示数量后停止。
                    return  # EN: Exit function. CN: 退出函数。


# ============================================================
# 9. Checkpoint utilities
# 9. Checkpoint 保存和加载工具
# ============================================================

def save_checkpoint(model, optimizer, epoch, loss_history, path):  # EN: Save model checkpoint. CN: 保存模型 checkpoint。
    os.makedirs(os.path.dirname(path), exist_ok=True)  # EN: Create checkpoint folder if needed. CN: 如果 checkpoint 文件夹不存在，则创建。
    torch.save(  # EN: Save checkpoint dictionary. CN: 保存 checkpoint 字典。
        {  # EN: Begin checkpoint dictionary. CN: 开始 checkpoint 字典。
            "epoch": epoch,  # EN: Save current epoch. CN: 保存当前 epoch。
            "model_state_dict": model.state_dict(),  # EN: Save model parameters. CN: 保存模型参数。
            "optimizer_state_dict": optimizer.state_dict(),  # EN: Save optimizer state. CN: 保存优化器状态。
            "loss_history": loss_history,  # EN: Save training loss history. CN: 保存训练 loss 历史。
            "num_classes": NUM_CLASSES,  # EN: Save class number. CN: 保存类别数。
            "class_names": CLASS_NAMES,  # EN: Save class names. CN: 保存类别名称。
        },  # EN: End checkpoint dictionary. CN: 结束 checkpoint 字典。
        path  # EN: Save path. CN: 保存路径。
    )  # EN: End torch.save. CN: 结束 torch.save。
    print(f"Checkpoint saved to: {path}")  # EN: Print checkpoint path. CN: 打印 checkpoint 路径。


def load_checkpoint(path, device):  # EN: Load a saved checkpoint. CN: 加载保存好的 checkpoint。
    checkpoint = torch.load(path, map_location=device)  # EN: Load checkpoint to selected device. CN: 将 checkpoint 加载到指定设备。
    model = build_model(num_classes=checkpoint["num_classes"])  # EN: Rebuild model with saved class number. CN: 根据保存的类别数重建模型。
    model.load_state_dict(checkpoint["model_state_dict"])  # EN: Load model weights. CN: 加载模型权重。
    model.to(device)  # EN: Move model to device. CN: 将模型移动到设备。
    model.eval()  # EN: Set model to evaluation mode. CN: 将模型设置为评估模式。
    print(f"Checkpoint loaded from: {path}")  # EN: Print checkpoint path. CN: 打印 checkpoint 路径。
    print(f"Class names: {checkpoint['class_names']}")  # EN: Print class names. CN: 打印类别名称。
    return model  # EN: Return loaded model. CN: 返回加载好的模型。


# ============================================================
# 10. Plot training loss
# 10. 绘制训练 loss
# ============================================================

def plot_loss(loss_history):  # EN: Plot training loss history. CN: 绘制训练 loss 曲线。
    plt.figure(figsize=(7, 5))  # EN: Create figure. CN: 创建画布。
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")  # EN: Plot average training loss per epoch. CN: 绘制每个 epoch 的平均训练 loss。
    plt.xlabel("Epoch")  # EN: Set x-axis label. CN: 设置 x 轴标签。
    plt.ylabel("Average Training Loss")  # EN: Set y-axis label. CN: 设置 y 轴标签。
    plt.title("Instance Segmentation Training Loss")  # EN: Set plot title. CN: 设置图标题。
    plt.grid(True)  # EN: Show grid. CN: 显示网格。
    plt.show()  # EN: Display plot. CN: 显示曲线。


# ============================================================
# 11. Main function
# 11. 主函数
# ============================================================

def main():  # EN: Define main training and inference pipeline. CN: 定义主训练和推理流程。
    set_seed(SEED)  # EN: Set random seed. CN: 设置随机种子。
    device = get_device()  # EN: Select CPU or GPU. CN: 选择 CPU 或 GPU。
    dataset_dir = download_and_extract_dataset()  # EN: Download or find dataset directory. CN: 下载或获取数据集目录。
    full_dataset = PennFudanSegmentationDataset(root=dataset_dir, transforms=get_transform(train=True))  # EN: Create full training-style dataset. CN: 创建带训练 transform 的完整数据集。
    test_dataset = PennFudanSegmentationDataset(root=dataset_dir, transforms=get_transform(train=False))  # EN: Create validation-style dataset. CN: 创建带验证 transform 的数据集。
    indices = torch.randperm(len(full_dataset)).tolist()  # EN: Randomly shuffle dataset indices. CN: 随机打乱数据索引。
    train_size = int(TRAIN_RATIO * len(indices))  # EN: Compute training set size. CN: 计算训练集大小。
    train_indices = indices[:train_size]  # EN: Select training indices. CN: 选择训练索引。
    test_indices = indices[train_size:]  # EN: Select testing indices. CN: 选择测试索引。
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)  # EN: Build training subset. CN: 构建训练子集。
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)  # EN: Build testing subset. CN: 构建测试子集。
    train_loader = torch.utils.data.DataLoader(  # EN: Create training dataloader. CN: 创建训练 DataLoader。
        train_dataset,  # EN: Training subset. CN: 训练子集。
        batch_size=BATCH_SIZE,  # EN: Training batch size. CN: 训练 batch size。
        shuffle=True,  # EN: Shuffle training data. CN: 打乱训练数据。
        num_workers=NUM_WORKERS,  # EN: Number of dataloader workers. CN: DataLoader worker 数量。
        collate_fn=collate_fn  # EN: Custom collate for variable-size targets. CN: 用自定义 collate 处理不同数量目标。
    )  # EN: End training dataloader. CN: 结束训练 DataLoader。
    test_loader = torch.utils.data.DataLoader(  # EN: Create testing dataloader. CN: 创建测试 DataLoader。
        test_dataset,  # EN: Testing subset. CN: 测试子集。
        batch_size=1,  # EN: Use batch size 1 for easy visualization. CN: 使用 batch size 1 方便可视化。
        shuffle=False,  # EN: Do not shuffle testing data. CN: 测试集不打乱。
        num_workers=NUM_WORKERS,  # EN: Number of dataloader workers. CN: DataLoader worker 数量。
        collate_fn=collate_fn  # EN: Custom collate for detection/segmentation targets. CN: 用自定义 collate 处理检测/分割 target。
    )  # EN: End testing dataloader. CN: 结束测试 DataLoader。
    print(f"Training images: {len(train_dataset)}")  # EN: Print number of training images. CN: 打印训练图像数量。
    print(f"Testing images: {len(test_dataset)}")  # EN: Print number of testing images. CN: 打印测试图像数量。
    model = build_model(num_classes=NUM_CLASSES)  # EN: Build Mask R-CNN model. CN: 构建 Mask R-CNN 模型。
    model.to(device)  # EN: Move model to selected device. CN: 将模型移动到指定设备。
    params = [p for p in model.parameters() if p.requires_grad]  # EN: Collect trainable parameters. CN: 收集需要训练的参数。
    optimizer = torch.optim.SGD(  # EN: Create SGD optimizer. CN: 创建 SGD 优化器。
        params,  # EN: Trainable parameters. CN: 需要训练的参数。
        lr=LEARNING_RATE,  # EN: Learning rate. CN: 学习率。
        momentum=MOMENTUM,  # EN: Momentum. CN: 动量。
        weight_decay=WEIGHT_DECAY  # EN: Weight decay. CN: 权重衰减。
    )  # EN: End optimizer definition. CN: 结束优化器定义。
    scheduler = torch.optim.lr_scheduler.StepLR(  # EN: Create learning-rate scheduler. CN: 创建学习率调度器。
        optimizer,  # EN: Optimizer to schedule. CN: 需要调度的优化器。
        step_size=LR_STEP_SIZE,  # EN: Decay interval. CN: 衰减间隔。
        gamma=LR_GAMMA  # EN: Decay factor. CN: 衰减系数。
    )  # EN: End scheduler definition. CN: 结束调度器定义。
    loss_history = []  # EN: Create list to store epoch losses. CN: 创建列表保存每个 epoch 的 loss。
    for epoch in range(1, NUM_EPOCHS + 1):  # EN: Loop over training epochs. CN: 遍历训练 epoch。
        avg_loss = train_one_epoch(  # EN: Train one epoch. CN: 训练一个 epoch。
            model=model,  # EN: Mask R-CNN model. CN: Mask R-CNN 模型。
            optimizer=optimizer,  # EN: Optimizer. CN: 优化器。
            dataloader=train_loader,  # EN: Training dataloader. CN: 训练 DataLoader。
            device=device,  # EN: Selected device. CN: 当前设备。
            epoch=epoch  # EN: Current epoch index. CN: 当前 epoch 编号。
        )  # EN: End one-epoch training call. CN: 结束单 epoch 训练调用。
        loss_history.append(avg_loss)  # EN: Save epoch average loss. CN: 保存当前 epoch 平均 loss。
        scheduler.step()  # EN: Update learning rate. CN: 更新学习率。
        save_checkpoint(  # EN: Save checkpoint after each epoch. CN: 每个 epoch 后保存 checkpoint。
            model=model,  # EN: Model to save. CN: 要保存的模型。
            optimizer=optimizer,  # EN: Optimizer to save. CN: 要保存的优化器。
            epoch=epoch,  # EN: Current epoch. CN: 当前 epoch。
            loss_history=loss_history,  # EN: Loss history. CN: loss 历史。
            path=CHECKPOINT_PATH  # EN: Checkpoint path. CN: checkpoint 路径。
        )  # EN: End checkpoint saving. CN: 结束 checkpoint 保存。
    plot_loss(loss_history)  # EN: Plot training loss. CN: 绘制训练 loss 曲线。
    run_inference(  # EN: Run instance segmentation inference. CN: 运行实例分割推理。
        model=model,  # EN: Trained model. CN: 训练好的模型。
        dataloader=test_loader,  # EN: Testing dataloader. CN: 测试 DataLoader。
        device=device,  # EN: Selected device. CN: 当前设备。
        score_threshold=SCORE_THRESHOLD,  # EN: Score threshold. CN: 置信度阈值。
        mask_threshold=MASK_THRESHOLD,  # EN: Mask threshold. CN: mask 二值化阈值。
        max_images=3  # EN: Number of images to visualize. CN: 可视化图像数量。
    )  # EN: End inference. CN: 结束推理。


if __name__ == "__main__":  # EN: Run main only when this file is executed directly. CN: 只有直接运行该文件时才执行 main。
    main()  # EN: Start the full segmentation pipeline. CN: 启动完整实例分割流程。
