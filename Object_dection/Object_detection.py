# ============================================================
# Object Detection Template: Faster R-CNN + ResNet50-FPN
# 目标检测模板：Faster R-CNN + ResNet50-FPN
#
# This template is suitable for coding interviews and custom
# object detection tasks.
# 该模板适合 coding interview 和自定义目标检测任务。
# ============================================================


# ============================================================
# 0. Import libraries
# 0. 导入所需库
# ============================================================

import os
# EN: Used for file path operations, checking folders, and joining paths.
# CN: 用于文件路径操作、检查文件夹是否存在、拼接路径。

import time
# EN: Used to measure training time.
# CN: 用于统计训练时间。

import zipfile
# EN: Used to extract downloaded zip files.
# CN: 用于解压下载的数据集压缩包。

import random
# EN: Used for random seed control and dataset splitting.
# CN: 用于控制随机种子和数据集随机划分。

import urllib.request
# EN: Used to download the dataset from a URL.
# CN: 用于从网络链接下载数据集。

import torch
# EN: Core PyTorch library.
# CN: PyTorch 核心库。

import torchvision
# EN: Computer vision library built on PyTorch. Provides models, datasets, transforms, and utilities.
# CN: 基于 PyTorch 的计算机视觉库，提供模型、数据集、图像变换和工具函数。

import torch.utils.data
# EN: Provides Dataset, DataLoader, and Subset utilities.
# CN: 提供 Dataset、DataLoader 和 Subset 等数据加载工具。

from PIL import Image
# EN: Used to open and convert image files.
# CN: 用于读取和转换图像文件。

import numpy as np
# EN: Used for numerical operations, especially mask processing.
# CN: 用于数值计算，尤其是处理 mask。

import matplotlib.pyplot as plt
# EN: Used for plotting images, bounding boxes, and loss curves.
# CN: 用于绘制图像、bounding boxes 和 loss 曲线。

from torchvision.models.detection import fasterrcnn_resnet50_fpn
# EN: Imports Faster R-CNN with ResNet50-FPN backbone.
# CN: 导入 Faster R-CNN 模型，backbone 是 ResNet50-FPN。

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# EN: Used to replace the final detection classification head.
# CN: 用于替换 Faster R-CNN 最后的检测分类头。

from torchvision.transforms import v2 as T
# EN: Imports torchvision v2 transforms. These can transform both images and targets.
# CN: 导入 torchvision v2 图像变换工具，可以同时处理 image 和 target。
# NOTE:
# EN: If your torchvision version does not support v2, replace this line with:
#     import torchvision.transforms as T
# CN: 如果你的 torchvision 版本不支持 v2，可以改成：
#     import torchvision.transforms as T

from torchvision.utils import draw_bounding_boxes
# EN: Utility function to draw predicted bounding boxes on images.
# CN: 用于在图像上绘制预测 bounding boxes。


# ============================================================
# 1. Configuration
# 1. 参数配置
# ============================================================

DATA_ROOT = "data"
# EN: Root folder where the dataset will be downloaded and extracted.
# CN: 数据集下载和解压的根目录。

DATASET_NAME = "PennFudanPed"
# EN: Name of the dataset folder after extraction.
# CN: 数据集解压后的文件夹名称。

DATASET_ZIP_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
# EN: Download URL for the Penn-Fudan Pedestrian dataset.
# CN: Penn-Fudan 行人检测数据集的下载链接。

OUTPUT_DIR = "outputs_detection"
# EN: Folder used to save model checkpoints.
# CN: 用于保存模型 checkpoint 的文件夹。

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "fasterrcnn_resnet50_fpn_pennfudan.pth"
)
# EN: Full path of the saved model checkpoint.
# CN: 模型 checkpoint 的完整保存路径。

NUM_CLASSES = 2
# EN: Number of classes for Faster R-CNN.
#     Important: torchvision detection models require background class.
#     Here:
#       class 0 = background
#       class 1 = pedestrian
# CN: Faster R-CNN 的类别数量。
#     注意：torchvision detection 模型需要包含 background 类。
#     这里：
#       类别 0 = background
#       类别 1 = pedestrian

BATCH_SIZE = 2
# EN: Batch size. Detection models usually require small batch size due to GPU memory.
# CN: batch size。目标检测模型显存占用较大，通常 batch size 设置较小。

NUM_EPOCHS = 5
# EN: Number of training epochs.
# CN: 训练轮数。

LEARNING_RATE = 0.005
# EN: Initial learning rate for SGD optimizer.
# CN: SGD 优化器的初始学习率。

MOMENTUM = 0.9
# EN: Momentum improves SGD convergence stability.
# CN: 动量项可以提升 SGD 的收敛稳定性。

WEIGHT_DECAY = 0.0005
# EN: L2 regularization to reduce overfitting.
# CN: L2 正则化，用于减少过拟合。

LR_STEP_SIZE = 3
# EN: Reduce learning rate every 3 epochs.
# CN: 每 3 个 epoch 降低一次学习率。

LR_GAMMA = 0.1
# EN: Learning rate will be multiplied by 0.1 when scheduler steps.
# CN: 每次学习率调整时，将 learning rate 乘以 0.1。

NUM_WORKERS = 0
# EN: Number of subprocesses for data loading.
#     Use 0 on Windows to avoid multiprocessing errors.
# CN: DataLoader 加载数据时使用的子进程数量。
#     Windows 下建议设置为 0，避免多进程报错。

TRAIN_RATIO = 0.8
# EN: Use 80% data for training and 20% for testing.
# CN: 使用 80% 数据作为训练集，20% 作为测试集。

SEED = 42
# EN: Random seed for reproducibility.
# CN: 随机种子，用于保证结果可复现。

SCORE_THRESHOLD = 0.5
# EN: During inference, only predictions with confidence >= 0.5 are displayed.
# CN: 推理时，只显示置信度大于等于 0.5 的预测框。


# ============================================================
# 2. Device and reproducibility
# 2. 设备选择和随机种子设置
# ============================================================

def set_seed(seed=42):
    # EN: Set Python random seed.
    # CN: 设置 Python 内置 random 模块的随机种子。
    random.seed(seed)

    # EN: Set NumPy random seed.
    # CN: 设置 NumPy 随机种子。
    np.random.seed(seed)

    # EN: Set PyTorch CPU random seed.
    # CN: 设置 PyTorch CPU 随机种子。
    torch.manual_seed(seed)

    # EN: Set PyTorch CUDA random seed for all GPUs.
    # CN: 设置所有 GPU 的 CUDA 随机种子。
    torch.cuda.manual_seed_all(seed)


def get_device():
    # EN: Use GPU if CUDA is available; otherwise use CPU.
    # CN: 如果 CUDA 可用则使用 GPU，否则使用 CPU。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # EN: Print selected device.
    # CN: 打印当前使用的设备。
    print(f"Using device: {device}")

    # EN: If using GPU, print GPU name.
    # CN: 如果使用 GPU，打印 GPU 名称。
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # EN: Return device object for later use.
    # CN: 返回 device，后续用于把模型和数据移动到 GPU/CPU。
    return device


# ============================================================
# 3. Download and extract dataset
# 3. 下载并解压数据集
# ============================================================

def download_and_extract_dataset():
    """
    EN:
        Download Penn-Fudan Pedestrian Dataset if it does not exist.

        Expected structure after extraction:

        data/
            PennFudanPed/
                PNGImages/
                    FudanPed00001.png
                    ...
                PedMasks/
                    FudanPed00001_mask.png
                    ...

    CN:
        如果 Penn-Fudan 数据集不存在，则自动下载并解压。

        解压后的目录结构应为：

        data/
            PennFudanPed/
                PNGImages/
                    FudanPed00001.png
                    ...
                PedMasks/
                    FudanPed00001_mask.png
                    ...
    """

    # EN: Create the root data folder if it does not exist.
    # CN: 如果 data 根目录不存在，则创建它。
    os.makedirs(DATA_ROOT, exist_ok=True)

    # EN: Dataset directory path, e.g. data/PennFudanPed.
    # CN: 数据集目录路径，例如 data/PennFudanPed。
    dataset_dir = os.path.join(DATA_ROOT, DATASET_NAME)

    # EN: Local path of downloaded zip file.
    # CN: 下载后的 zip 文件本地保存路径。
    zip_path = os.path.join(DATA_ROOT, "PennFudanPed.zip")

    # EN: If dataset already exists, skip download.
    # CN: 如果数据集已经存在，则跳过下载。
    if os.path.exists(dataset_dir):
        print(f"Dataset already exists: {dataset_dir}")
        return dataset_dir

    # EN: Print download message.
    # CN: 打印下载提示。
    print("Downloading dataset...")

    # EN: Print dataset URL.
    # CN: 打印数据集下载链接。
    print(f"URL: {DATASET_ZIP_URL}")

    # EN: Download dataset zip file.
    # CN: 下载数据集压缩包。
    urllib.request.urlretrieve(DATASET_ZIP_URL, zip_path)

    # EN: Print extraction message.
    # CN: 打印解压提示。
    print("Extracting dataset...")

    # EN: Open zip file in read mode.
    # CN: 以只读模式打开 zip 文件。
    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        # EN: Extract all files to DATA_ROOT.
        # CN: 将所有文件解压到 DATA_ROOT 目录。
        zip_ref.extractall(DATA_ROOT)

    # EN: Print final dataset path.
    # CN: 打印数据集解压后的路径。
    print(f"Dataset extracted to: {dataset_dir}")

    # EN: Return dataset directory.
    # CN: 返回数据集目录路径。
    return dataset_dir


# ============================================================
# 4. Transform functions
# 4. 图像预处理函数
# ============================================================

def get_transform(train):
    """
    EN:
        Build transform pipeline for training or testing.

        For object detection, transforms should ideally modify both
        the image and the target boxes.

        With torchvision v2:
        - RandomHorizontalFlip can update bounding boxes automatically.
        - ToImage converts PIL image to image tensor.
        - ToDtype converts image to float32 and scales to [0, 1].

    CN:
        构建训练或测试阶段的图像预处理流程。

        对于目标检测任务，transform 最好能同时修改 image 和 target boxes。

        使用 torchvision v2 时：
        - RandomHorizontalFlip 可以自动同步更新 bounding boxes。
        - ToImage 将 PIL 图像转换为 Tensor 图像格式。
        - ToDtype 将图像转换为 float32，并缩放到 [0, 1]。
    """

    # EN: Create an empty list for transforms.
    # CN: 创建一个空列表，用于存放 transform 操作。
    transforms = []

    # EN: If this is training mode, add random augmentation.
    # CN: 如果是训练模式，加入随机数据增强。
    if train:

        # EN: Randomly flip image and target horizontally with 50% probability.
        # CN: 以 50% 概率随机水平翻转图像和对应 target。
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    # EN: Convert PIL image to torchvision image tensor.
    # CN: 将 PIL 图像转换为 torchvision image tensor。
    transforms.append(T.ToImage())

    # EN: Convert image to float32 and scale pixel values to [0, 1].
    # CN: 将图像转换为 float32，并把像素值缩放到 [0, 1]。
    transforms.append(T.ToDtype(torch.float32, scale=True))

    # EN: Combine all transforms into one pipeline.
    # CN: 将所有 transform 组合成一个完整流程。
    return T.Compose(transforms)


# ============================================================
# 5. Custom dataset
# 5. 自定义目标检测数据集
# ============================================================

class PennFudanDataset(torch.utils.data.Dataset):
    """
    EN:
        Custom Dataset for Penn-Fudan pedestrian detection.

        This dataset provides:
        - RGB image
        - instance mask image

        We convert each instance mask into a bounding box.

    CN:
        Penn-Fudan 行人检测数据集的自定义 Dataset。

        这个数据集提供：
        - RGB 图像
        - instance mask 图像

        我们将每个 instance mask 转换成 bounding box。
    """

    def __init__(self, root, transforms=None):
        # EN: Save dataset root directory.
        # CN: 保存数据集根目录。
        self.root = root

        # EN: Save transform pipeline.
        # CN: 保存图像预处理流程。
        self.transforms = transforms

        # EN: Get all image file names from PNGImages folder.
        # CN: 从 PNGImages 文件夹中读取所有图像文件名。
        self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))

        # EN: Get all mask file names from PedMasks folder.
        # CN: 从 PedMasks 文件夹中读取所有 mask 文件名。
        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))

    def __getitem__(self, idx):
        # EN: Build image path according to index.
        # CN: 根据 index 拼接图像路径。
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])

        # EN: Build mask path according to index.
        # CN: 根据 index 拼接 mask 路径。
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        # EN: Open image and convert it to RGB format.
        # CN: 读取图像，并转换为 RGB 格式。
        img = Image.open(img_path).convert("RGB")

        # EN: Open corresponding mask image.
        # CN: 读取对应的 mask 图像。
        mask = Image.open(mask_path)

        # EN: Convert mask from PIL image to NumPy array.
        # CN: 将 PIL mask 转换成 NumPy 数组。
        mask = np.array(mask)

        # EN: Get unique pixel values in the mask.
        #     Each non-zero value corresponds to one pedestrian instance.
        # CN: 获取 mask 中所有唯一像素值。
        #     每个非零像素值对应一个行人 instance。
        obj_ids = np.unique(mask)

        # EN: Remove background id 0.
        # CN: 去掉 background 的 id 0。
        obj_ids = obj_ids[1:]

        # EN: Create binary masks for each object instance.
        #     Shape: [num_objects, height, width]
        # CN: 为每个目标 instance 创建二值 mask。
        #     形状为：[目标数量, 高度, 宽度]
        masks = mask == obj_ids[:, None, None]

        # EN: Create an empty list to store bounding boxes.
        # CN: 创建空列表，用于保存 bounding boxes。
        boxes = []

        # EN: Loop over every object instance.
        # CN: 遍历每一个目标 instance。
        for i in range(len(obj_ids)):

            # EN: Find all pixel positions where this object mask is True.
            # CN: 找到当前目标 mask 中所有为 True 的像素位置。
            pos = np.where(masks[i])

            # EN: Minimum x coordinate of bounding box.
            # CN: bounding box 的最小 x 坐标。
            xmin = np.min(pos[1])

            # EN: Maximum x coordinate of bounding box.
            # CN: bounding box 的最大 x 坐标。
            xmax = np.max(pos[1])

            # EN: Minimum y coordinate of bounding box.
            # CN: bounding box 的最小 y 坐标。
            ymin = np.min(pos[0])

            # EN: Maximum y coordinate of bounding box.
            # CN: bounding box 的最大 y 坐标。
            ymax = np.max(pos[0])

            # EN: Append box in [xmin, ymin, xmax, ymax] format.
            # CN: 以 [xmin, ymin, xmax, ymax] 格式保存 box。
            boxes.append([xmin, ymin, xmax, ymax])

        # EN: Convert boxes list to float32 Tensor.
        #     Shape: [num_objects, 4]
        # CN: 将 boxes 列表转换为 float32 Tensor。
        #     形状为：[目标数量, 4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # EN: Create labels. All objects are pedestrians, so all labels are 1.
        # CN: 创建类别标签。所有目标都是 pedestrian，所以标签全是 1。
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)

        # EN: Convert masks to uint8 Tensor.
        # CN: 将 masks 转换为 uint8 Tensor。
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # EN: Create image id tensor.
        # CN: 创建 image id Tensor。
        image_id = torch.tensor([idx])

        # EN: Calculate area of each bounding box.
        #     area = height * width
        # CN: 计算每个 bounding box 的面积。
        #     面积 = 高度 × 宽度
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # EN: iscrowd indicates whether an object is a crowd region.
        #     Here all objects are not crowd, so set to 0.
        # CN: iscrowd 表示目标是否是 crowd 区域。
        #     这里所有目标都不是 crowd，所以设置为 0。
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        # EN: Build target dictionary required by torchvision detection models.
        # CN: 构建 torchvision detection 模型所需要的 target 字典。
        target = {
            "boxes": boxes,
            # EN: Bounding boxes in [xmin, ymin, xmax, ymax].
            # CN: bounding boxes，格式为 [xmin, ymin, xmax, ymax]。

            "labels": labels,
            # EN: Class labels for each box.
            # CN: 每个 box 对应的类别标签。

            "masks": masks,
            # EN: Instance masks. Useful for Mask R-CNN, optional for Faster R-CNN.
            # CN: instance masks。对 Mask R-CNN 有用；对 Faster R-CNN 不是必须。

            "image_id": image_id,
            # EN: Unique image id.
            # CN: 图像编号。

            "area": area,
            # EN: Area of each bounding box.
            # CN: 每个 bounding box 的面积。

            "iscrowd": iscrowd,
            # EN: Crowd label required for COCO-style evaluation.
            # CN: COCO 风格评估中需要的 crowd 标记。
        }

        # EN: Apply transforms to image and target if provided.
        # CN: 如果提供了 transforms，则对 image 和 target 进行变换。
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # EN: Return image and target.
        # CN: 返回图像和对应目标标注。
        return img, target

    def __len__(self):
        # EN: Return number of images in the dataset.
        # CN: 返回数据集中图像数量。
        return len(self.imgs)


# ============================================================
# 6. Collate function
# 6. 自定义 batch 组合函数
# ============================================================

def collate_fn(batch):
    """
    EN:
        Object detection images may contain different numbers of objects.
        Therefore, each target can have different tensor shapes.

        The default DataLoader collate function cannot stack them directly.
        So we return tuple(zip(*batch)).

    CN:
        目标检测中，每张图像包含的目标数量可能不同。
        因此每张图像的 target tensor 形状也可能不同。

        默认 DataLoader collate function 无法直接 stack 它们。
        所以这里返回 tuple(zip(*batch))。
    """

    # EN: Converts a list of (image, target) into:
    #     images = tuple(image1, image2, ...)
    #     targets = tuple(target1, target2, ...)
    # CN: 将 [(image, target), ...] 转换成：
    #     images = (image1, image2, ...)
    #     targets = (target1, target2, ...)
    return tuple(zip(*batch))


# ============================================================
# 7. Build Faster R-CNN model
# 7. 构建 Faster R-CNN 模型
# ============================================================

def build_model(num_classes):
    """
    EN:
        Build Faster R-CNN with ResNet50-FPN backbone.

        Transfer learning strategy:
        - Load pretrained Faster R-CNN trained on COCO.
        - Replace the final box predictor.
        - The new predictor outputs num_classes categories.

    CN:
        构建 Faster R-CNN + ResNet50-FPN 模型。

        迁移学习策略：
        - 加载在 COCO 上预训练好的 Faster R-CNN。
        - 替换最后的 box predictor。
        - 新 predictor 输出当前任务的类别数。
    """

    # EN: Load Faster R-CNN with pretrained COCO weights.
    # CN: 加载 COCO 上预训练好的 Faster R-CNN。
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # EN: Get input feature dimension of the original classification head.
    # CN: 获取原始分类头输入特征维度。
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # EN: Replace the predictor head with a new one for our num_classes.
    # CN: 用新的 predictor 替换原始 predictor，以适配当前类别数。
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    # EN: Return modified model.
    # CN: 返回修改后的模型。
    return model


# ============================================================
# 8. Train one epoch
# 8. 训练一个 epoch
# ============================================================

def train_one_epoch(model, optimizer, dataloader, device, epoch):
    # EN: Set model to training mode.
    # CN: 将模型设置为训练模式。
    model.train()

    # EN: Initialize total loss for this epoch.
    # CN: 初始化当前 epoch 的总 loss。
    total_loss = 0.0

    # EN: Record start time.
    # CN: 记录开始时间。
    start_time = time.time()

    # EN: Iterate over all batches.
    # CN: 遍历所有 batch。
    for batch_idx, (images, targets) in enumerate(dataloader):

        # EN: Move each image tensor to GPU/CPU.
        # CN: 将每张图像移动到 GPU/CPU。
        images = [img.to(device) for img in images]

        # EN: Move every tensor inside target dictionary to GPU/CPU.
        # CN: 将 target 字典中的每个 Tensor 移动到 GPU/CPU。
        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        # EN:
        #   In training mode, torchvision detection model returns a loss dictionary.
        #   Example keys:
        #       loss_classifier
        #       loss_box_reg
        #       loss_objectness
        #       loss_rpn_box_reg
        #
        # CN:
        #   在训练模式下，torchvision detection 模型会返回一个 loss 字典。
        #   常见 key 包括：
        #       loss_classifier：分类损失
        #       loss_box_reg：box 回归损失
        #       loss_objectness：RPN objectness 损失
        #       loss_rpn_box_reg：RPN box 回归损失
        loss_dict = model(images, targets)

        # EN: Sum all detection loss terms into one scalar loss.
        # CN: 将所有 detection loss 相加，得到一个总 loss。
        losses = sum(loss for loss in loss_dict.values())

        # EN: Clear old gradients.
        # CN: 清空上一轮残留梯度。
        optimizer.zero_grad()

        # EN: Backpropagate loss to compute gradients.
        # CN: 反向传播，计算梯度。
        losses.backward()

        # EN: Update model parameters.
        # CN: 使用优化器更新模型参数。
        optimizer.step()

        # EN: Accumulate training loss.
        # CN: 累加当前 batch 的 loss。
        total_loss += losses.item()

        # EN: Print training status every 10 batches.
        # CN: 每 10 个 batch 打印一次训练状态。
        if batch_idx % 10 == 0:

            # EN: Convert each loss tensor to a rounded Python number for printing.
            # CN: 将 loss 字典中的 Tensor 转换成保留 4 位小数的数字，方便打印。
            loss_items = {k: round(v.item(), 4) for k, v in loss_dict.items()}

            # EN: Print epoch, batch index, total loss, and individual loss terms.
            # CN: 打印 epoch、batch 编号、总 loss 和各个子 loss。
            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {losses.item():.4f} "
                f"{loss_items}"
            )

    # EN: Compute average loss over all batches.
    # CN: 计算当前 epoch 的平均 loss。
    avg_loss = total_loss / len(dataloader)

    # EN: Compute elapsed time.
    # CN: 计算当前 epoch 用时。
    elapsed = time.time() - start_time

    # EN: Print average loss.
    # CN: 打印平均 loss。
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

    # EN: Print epoch training time.
    # CN: 打印当前 epoch 训练时间。
    print(f"Epoch [{epoch}] Time: {elapsed:.1f}s")

    # EN: Return average loss for loss curve.
    # CN: 返回平均 loss，用于后续绘制 loss 曲线。
    return avg_loss


# ============================================================
# 9. Simple evaluation / inference
# 9. 简单推理和可视化评估
# ============================================================

@torch.no_grad()
# EN: Disable gradient computation for the whole inference function.
# CN: 对整个推理函数关闭梯度计算，节省显存和计算时间。
def run_inference(model, dataloader, device, score_threshold=0.5, max_images=3):
    """
    EN:
        Simple visual evaluation function.

        For formal object detection evaluation, use COCO mAP.
        For coding interviews, visual predictions are often enough.

    CN:
        简单的可视化评估函数。

        如果要正式评估目标检测模型，应该使用 COCO mAP。
        但在 coding interview 中，可视化预测框通常已经足够展示结果。
    """

    # EN: Set model to evaluation mode.
    # CN: 将模型设置为评估模式。
    model.eval()

    # EN: Count how many images have been visualized.
    # CN: 记录已经显示了多少张图像。
    images_shown = 0

    # EN: Iterate over test dataloader.
    # CN: 遍历测试集 DataLoader。
    for images, targets in dataloader:

        # EN: Move images to GPU/CPU.
        # CN: 将图像移动到 GPU/CPU。
        images_gpu = [img.to(device) for img in images]

        # EN:
        #   In eval mode, detection model returns prediction outputs.
        #   Each output contains:
        #       boxes, labels, scores
        #
        # CN:
        #   在 eval 模式下，detection 模型返回预测结果。
        #   每个 output 包含：
        #       boxes：预测框
        #       labels：预测类别
        #       scores：置信度
        outputs = model(images_gpu)

        # EN: Loop over every image and its output.
        # CN: 遍历每张图像和对应的预测输出。
        for img, output, target in zip(images, outputs, targets):

            # EN: Convert image from float [0, 1] to uint8 [0, 255].
            # CN: 将图像从 float [0, 1] 转换为 uint8 [0, 255]。
            img_uint8 = (img * 255).to(torch.uint8)

            # EN: Get prediction confidence scores and move them to CPU.
            # CN: 获取预测置信度，并移动到 CPU。
            scores = output["scores"].cpu()

            # EN: Keep only boxes whose score is above threshold.
            # CN: 只保留置信度大于阈值的预测框。
            keep = scores >= score_threshold

            # EN: Select filtered predicted boxes.
            # CN: 筛选后的预测框。
            boxes = output["boxes"].cpu()[keep]

            # EN: Select filtered predicted labels.
            # CN: 筛选后的预测类别。
            labels = output["labels"].cpu()[keep]

            # EN: Select filtered confidence scores.
            # CN: 筛选后的置信度。
            scores = scores[keep]

            # EN: Create text labels for visualization.
            # CN: 创建可视化显示用的标签文本。
            label_texts = [
                f"pedestrian {score:.2f}"
                for score in scores
            ]

            # EN: Draw bounding boxes on the image.
            # CN: 在图像上绘制 bounding boxes。
            drawn = draw_bounding_boxes(
                image=img_uint8,
                boxes=boxes,
                labels=label_texts,
                width=3
            )

            # EN: Create a figure.
            # CN: 创建图像窗口。
            plt.figure(figsize=(8, 6))

            # EN: Convert image from [C, H, W] to [H, W, C] for matplotlib.
            # CN: 将图像从 [C, H, W] 转成 [H, W, C] 以便 matplotlib 显示。
            plt.imshow(drawn.permute(1, 2, 0))

            # EN: Hide axis.
            # CN: 隐藏坐标轴。
            plt.axis("off")

            # EN: Set plot title.
            # CN: 设置图像标题。
            plt.title("Predicted bounding boxes")

            # EN: Show image.
            # CN: 显示图像。
            plt.show()

            # EN: Increase visualization counter.
            # CN: 已显示图像数量加 1。
            images_shown += 1

            # EN: Stop after visualizing max_images.
            # CN: 显示达到 max_images 张后停止。
            if images_shown >= max_images:
                return


# ============================================================
# 10. Save and load model
# 10. 保存和加载模型
# ============================================================

def save_checkpoint(model, optimizer, epoch, loss_history, path):
    # EN: Create output directory if it does not exist.
    # CN: 如果输出目录不存在，则创建它。
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # EN: Save model state, optimizer state, epoch, loss history, and class info.
    # CN: 保存模型参数、优化器状态、epoch、loss 历史和类别信息。
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "num_classes": NUM_CLASSES,
            "class_names": ["background", "pedestrian"],
        },
        path
    )

    # EN: Print checkpoint save path.
    # CN: 打印 checkpoint 保存路径。
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(path, device):
    # EN: Load checkpoint from disk.
    # CN: 从磁盘读取 checkpoint。
    checkpoint = torch.load(path, map_location=device)

    # EN: Rebuild model with saved number of classes.
    # CN: 根据保存的类别数重新构建模型。
    model = build_model(num_classes=checkpoint["num_classes"])

    # EN: Load saved model weights.
    # CN: 加载保存的模型权重。
    model.load_state_dict(checkpoint["model_state_dict"])

    # EN: Move model to selected device.
    # CN: 将模型移动到指定设备。
    model.to(device)

    # EN: Set model to evaluation mode.
    # CN: 将模型设置为评估模式。
    model.eval()

    # EN: Print loading message.
    # CN: 打印加载成功信息。
    print(f"Checkpoint loaded from: {path}")

    # EN: Print class names.
    # CN: 打印类别名称。
    print(f"Class names: {checkpoint['class_names']}")

    # EN: Return loaded model.
    # CN: 返回加载好的模型。
    return model


# ============================================================
# 11. Plot training loss
# 11. 绘制训练 loss 曲线
# ============================================================

def plot_loss(loss_history):
    # EN: Create figure.
    # CN: 创建图像窗口。
    plt.figure(figsize=(7, 5))

    # EN: Plot epoch index versus average training loss.
    # CN: 绘制 epoch 编号与平均训练 loss 的关系曲线。
    plt.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        marker="o"
    )

    # EN: Set x-axis label.
    # CN: 设置 x 轴标签。
    plt.xlabel("Epoch")

    # EN: Set y-axis label.
    # CN: 设置 y 轴标签。
    plt.ylabel("Average Training Loss")

    # EN: Set plot title.
    # CN: 设置图像标题。
    plt.title("Object Detection Training Loss")

    # EN: Show grid.
    # CN: 显示网格线。
    plt.grid(True)

    # EN: Display plot.
    # CN: 显示曲线。
    plt.show()


# ============================================================
# 12. Main function
# 12. 主函数
# ============================================================

def main():
    # EN: Set random seeds.
    # CN: 设置随机种子。
    set_seed(SEED)

    # EN: Select GPU or CPU.
    # CN: 选择 GPU 或 CPU。
    device = get_device()

    # EN: Download and extract dataset if needed.
    # CN: 如果需要，则下载并解压数据集。
    dataset_dir = download_and_extract_dataset()

    # EN:
    #   Create full training dataset with training transforms.
    #   RandomHorizontalFlip may be applied here.
    #
    # CN:
    #   创建完整训练数据集，并使用训练阶段 transforms。
    #   这里可能会使用随机水平翻转增强。
    full_dataset = PennFudanDataset(
        root=dataset_dir,
        transforms=get_transform(train=True)
    )

    # EN:
    #   Create full test dataset with deterministic transforms.
    #   It uses the same images but different transform settings.
    #
    # CN:
    #   创建完整测试数据集，并使用测试阶段 transforms。
    #   它读取同一批图像，但 transform 设置不同。
    test_dataset = PennFudanDataset(
        root=dataset_dir,
        transforms=get_transform(train=False)
    )

    # EN: Randomly generate shuffled image indices.
    # CN: 随机生成打乱后的图像索引。
    indices = torch.randperm(len(full_dataset)).tolist()

    # EN: Compute number of training samples.
    # CN: 计算训练样本数量。
    train_size = int(TRAIN_RATIO * len(indices))

    # EN: First part is used for training.
    # CN: 前一部分索引用作训练集。
    train_indices = indices[:train_size]

    # EN: Remaining part is used for testing.
    # CN: 剩余索引用作测试集。
    test_indices = indices[train_size:]

    # EN: Create training subset.
    # CN: 创建训练子集。
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)

    # EN: Create testing subset.
    # CN: 创建测试子集。
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # EN: Create training DataLoader.
    # CN: 创建训练 DataLoader。
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # EN: Create testing DataLoader.
    # CN: 创建测试 DataLoader。
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # EN: Print number of training images.
    # CN: 打印训练图像数量。
    print(f"Training images: {len(train_dataset)}")

    # EN: Print number of testing images.
    # CN: 打印测试图像数量。
    print(f"Testing images: {len(test_dataset)}")

    # EN: Build Faster R-CNN model with required number of classes.
    # CN: 构建 Faster R-CNN 模型，并设置当前任务类别数。
    model = build_model(num_classes=NUM_CLASSES)

    # EN: Move model to GPU/CPU.
    # CN: 将模型移动到 GPU/CPU。
    model.to(device)

    # EN: Collect all trainable parameters.
    # CN: 收集所有需要训练的参数。
    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    # EN: Define SGD optimizer.
    # CN: 定义 SGD 优化器。
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # EN: Define learning rate scheduler.
    # CN: 定义学习率调度器。
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA
    )

    # EN: Create an empty list to store average loss of each epoch.
    # CN: 创建空列表，用于保存每个 epoch 的平均 loss。
    loss_history = []

    # EN: Training loop over epochs.
    # CN: 按 epoch 进行训练循环。
    for epoch in range(1, NUM_EPOCHS + 1):

        # EN: Train model for one epoch.
        # CN: 训练一个 epoch。
        avg_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            epoch=epoch
        )

        # EN: Save average loss to history.
        # CN: 将平均 loss 保存到历史记录中。
        loss_history.append(avg_loss)

        # EN: Update learning rate.
        # CN: 更新学习率。
        lr_scheduler.step()

        # EN: Save checkpoint after every epoch.
        # CN: 每个 epoch 后保存 checkpoint。
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss_history=loss_history,
            path=CHECKPOINT_PATH
        )

    # EN: Plot training loss curve.
    # CN: 绘制训练 loss 曲线。
    plot_loss(loss_history)

    # EN: Run visual inference on test images.
    # CN: 在测试图像上进行可视化推理。
    run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
        score_threshold=SCORE_THRESHOLD,
        max_images=3
    )


# ============================================================
# 13. Python entry point
# 13. Python 程序入口
# ============================================================

if __name__ == "__main__":
    # EN:
    #   This ensures main() only runs when this script is executed directly.
    #   It is especially important on Windows when using DataLoader.
    #
    # CN:
    #   这可以确保只有直接运行该脚本时才会执行 main()。
    #   在 Windows 使用 DataLoader 时尤其重要。
    main()