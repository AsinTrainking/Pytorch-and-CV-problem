# mini_vit_interview_template_bilingual.py  # EN: Mini Vision Transformer template for interview image classification. CN: 面试用最小 Vision Transformer 图像分类模板。

import os  
# EN: Import OS utilities for file paths. CN: 导入 os，用于处理文件路径。
import time  
# EN: Import time for measuring training duration. CN: 导入 time，用于统计训练时间。
import copy  
# EN: Import copy for saving the best model weights. CN: 导入 copy，用于保存最佳模型权重。
import random  
# EN: Import random for reproducibility. CN: 导入 random，用于设置随机种子。
import numpy as np  # EN: Import NumPy for numerical operations. CN: 导入 NumPy，用于数值计算。
import matplotlib.pyplot as plt  
# EN: Import matplotlib for visualization. CN: 导入 matplotlib，用于图像和曲线可视化。

import torch  
# EN: Import PyTorch core library. CN: 导入 PyTorch 核心库。
import torch.nn as nn  
# EN: Import neural network modules. CN: 导入神经网络模块。
import torch.optim as optim  
# EN: Import optimization algorithms. CN: 导入优化器模块。
from torch.optim import lr_scheduler  
# EN: Import learning-rate scheduler. CN: 导入学习率调度器。
from torch.utils.data import DataLoader  
# EN: Import DataLoader for mini-batch loading. CN: 导入 DataLoader，用于批量加载数据。
from torchvision import datasets, transforms  
# EN: Import ImageFolder dataset and image transforms. CN: 导入 ImageFolder 数据集和图像预处理。
from torchvision.utils import make_grid  
# EN: Import make_grid to show a batch of images. CN: 导入 make_grid，用于显示一批图片。
from PIL import Image  
# EN: Import PIL for single-image inference. CN: 导入 PIL，用于单张图片预测。


# ============================================================
# 1. Basic configuration
# 1. 基础配置
# ============================================================

DATA_DIR = "hymenoptera_data"  # EN: Dataset root folder. CN: 数据集根目录。
OUTPUT_DIR = "outputs_mini_vit"  # EN: Folder for checkpoints and plots. CN: 保存模型和曲线图的文件夹。
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_mini_vit.pth")  
# EN: Best checkpoint path. CN: 最佳模型保存路径。
HISTORY_PLOT_PATH = os.path.join(OUTPUT_DIR, "mini_vit_training_curves.png")  
# EN: Training curve path. CN: 训练曲线保存路径。

IMG_SIZE = 64  # EN: Input image size; CPU-friendly. CN: 输入图像大小；适合 CPU 演示。
PATCH_SIZE = 8  
# EN: Patch size; 64/8 gives 8x8=64 patches. CN: patch 大小；64/8 得到 8x8=64 个 patch。
IN_CHANNELS = 3  
# EN: RGB image has 3 channels. CN: RGB 图像有 3 个通道。
D_MODEL = 64  # EN: Transformer embedding dimension; small for CPU. CN: Transformer 特征维度；CPU 上使用较小值。
NHEAD = 4  # EN: Number of attention heads. CN: 多头注意力的 head 数量。
NUM_LAYERS = 1  # EN: Number of Transformer encoder layers; small for demo. CN: Transformer Encoder 层数；演示时用较小值。
DIM_FEEDFORWARD = 128  # EN: Hidden dimension in Transformer feed-forward network. CN: Transformer 前馈网络的隐藏层维度。
DROPOUT = 0.1  # EN: Dropout rate for regularization. CN: Dropout 比例，用于正则化。

BATCH_SIZE = 4  # EN: Small batch size for CPU/GPU compatibility. CN: 小 batch size，适合 CPU/GPU。
NUM_EPOCHS = 3  # EN: Few epochs for interview demonstration. CN: 面试演示时使用较少 epoch。
NUM_WORKERS = 0  # EN: Use 0 for Windows/Jupyter safety. CN: Windows/Jupyter 中建议设为 0。
LEARNING_RATE = 1e-3  # EN: Learning rate for Adam optimizer. CN: Adam 优化器的学习率。
WEIGHT_DECAY = 1e-4  # EN: Weight decay to reduce overfitting. CN: 权重衰减，用于减少过拟合。
STEP_SIZE = 2  # EN: Step size for learning-rate decay. CN: 学习率下降的间隔 epoch 数。
GAMMA = 0.5  # EN: Learning-rate decay factor. CN: 学习率衰减系数。
SEED = 42  # EN: Random seed for reproducibility. CN: 随机种子，保证结果尽量可复现。


# ============================================================
# 2. Reproducibility and device
# 2. 可复现性和设备选择
# ============================================================

def set_seed(seed=42):  
    # EN: Define a function to set random seeds. CN: 定义设置随机种子的函数。
    random.seed(seed)  
    # EN: Set Python random seed. CN: 设置 Python random 随机种子。
    np.random.seed(seed)  
    # EN: Set NumPy random seed. CN: 设置 NumPy 随机种子。
    torch.manual_seed(seed)  
    # EN: Set CPU random seed in PyTorch. CN: 设置 PyTorch CPU 随机种子。
    torch.cuda.manual_seed_all(seed)  
    # EN: Set GPU random seed if CUDA is available. CN: 如果有 GPU，设置 CUDA 随机种子。


def get_device():  
    # EN: Define a function to select CPU or GPU automatically. CN: 定义自动选择 CPU 或 GPU 的函数。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    # EN: Use GPU if available; otherwise CPU. CN: 如果有 GPU 就用 GPU，否则用 CPU。
    print(f"Using device: {device}")  
    # EN: Print selected device. CN: 打印当前使用的设备。
    if device.type == "cuda":  # EN: If the device is GPU. CN: 如果当前设备是 GPU。
        print(f"GPU name: {torch.cuda.get_device_name(0)}")  
        # EN: Print GPU name. CN: 打印 GPU 名称。
    return device  # EN: Return selected device. CN: 返回当前设备。


# ============================================================
# 3. Data preparation
# 3. 数据准备
# ============================================================

def get_data_transforms():  
    # EN: Define image preprocessing and augmentation. CN: 定义图像预处理和数据增强。
    data_transforms = {  
        # EN: Create transform dictionary for train and validation. CN: 创建训练和验证的预处理字典。
        "train": transforms.Compose([  # EN: Compose training transforms. CN: 组合训练集预处理。
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # EN: Resize all images to fixed size. CN: 将所有图像缩放到固定大小。
            transforms.RandomHorizontalFlip(p=0.5),  # EN: Randomly flip images for augmentation. CN: 随机水平翻转，用于数据增强。
            transforms.ToTensor(),  # EN: Convert PIL image to tensor in [0,1]. CN: 将 PIL 图像转成 [0,1] 范围的 Tensor。
            transforms.Normalize(  # EN: Normalize using ImageNet mean and std. CN: 使用 ImageNet 均值和方差进行归一化。
                mean=[0.485, 0.456, 0.406],  # EN: ImageNet channel mean. CN: ImageNet 每个通道的均值。
                std=[0.229, 0.224, 0.225]  # EN: ImageNet channel std. CN: ImageNet 每个通道的标准差。
            ),  # EN: End Normalize. CN: 结束 Normalize。
        ]),  # EN: End train transforms. CN: 结束训练集预处理。
        "val": transforms.Compose([  # EN: Compose validation transforms. CN: 组合验证集预处理。
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # EN: Resize validation images to fixed size. CN: 将验证图像缩放到固定大小。
            transforms.ToTensor(),  # EN: Convert image to tensor. CN: 将图像转成 Tensor。
            transforms.Normalize(  # EN: Use the same normalization as training. CN: 使用和训练集相同的归一化。
                mean=[0.485, 0.456, 0.406],  # EN: ImageNet channel mean. CN: ImageNet 通道均值。
                std=[0.229, 0.224, 0.225]  # EN: ImageNet channel std. CN: ImageNet 通道标准差。
            ),  # EN: End Normalize. CN: 结束 Normalize。
        ]),  # EN: End validation transforms. CN: 结束验证集预处理。
    }  # EN: End transform dictionary. CN: 结束预处理字典。
    return data_transforms  # EN: Return transforms. CN: 返回预处理方法。


def build_dataloaders(data_dir, batch_size, num_workers):  # EN: Build datasets and dataloaders. CN: 构建数据集和 DataLoader。
    data_transforms = get_data_transforms()  # EN: Get transform dictionary. CN: 获取预处理字典。
    image_datasets = {  # EN: Create ImageFolder datasets. CN: 创建 ImageFolder 数据集。
        phase: datasets.ImageFolder(  # EN: ImageFolder reads class names from folders. CN: ImageFolder 根据子文件夹自动识别类别。
            root=os.path.join(data_dir, phase),  # EN: Path such as hymenoptera_data/train. CN: 路径如 hymenoptera_data/train。
            transform=data_transforms[phase]  # EN: Apply corresponding transform. CN: 应用对应的预处理。
        )  # EN: End ImageFolder. CN: 结束 ImageFolder。
        for phase in ["train", "val"]  # EN: Build both train and validation datasets. CN: 同时构建训练集和验证集。
    }  # EN: End dataset dictionary. CN: 结束数据集字典。
    dataloaders = {  # EN: Create dataloaders. CN: 创建 DataLoader。
        phase: DataLoader(  # EN: DataLoader provides mini-batches. CN: DataLoader 提供小批量数据。
            image_datasets[phase],  # EN: Dataset for current phase. CN: 当前阶段的数据集。
            batch_size=batch_size,  # EN: Number of images per batch. CN: 每个 batch 的图像数量。
            shuffle=True if phase == "train" else False,  # EN: Shuffle training data only. CN: 只打乱训练集。
            num_workers=num_workers,  # EN: Number of worker processes. CN: 数据加载进程数量。
            pin_memory=True if torch.cuda.is_available() else False  # EN: Speed up GPU transfer if CUDA exists. CN: 如果有 CUDA，加速数据传输。
        )  # EN: End DataLoader. CN: 结束 DataLoader。
        for phase in ["train", "val"]  # EN: Build loaders for train and validation. CN: 构建训练和验证 DataLoader。
    }  # EN: End dataloader dictionary. CN: 结束 DataLoader 字典。
    dataset_sizes = {  # EN: Store dataset sizes. CN: 保存数据集大小。
        phase: len(image_datasets[phase])  # EN: Count images in each split. CN: 统计每个数据划分中的图像数量。
        for phase in ["train", "val"]  # EN: For train and validation. CN: 针对训练集和验证集。
    }  # EN: End dataset sizes. CN: 结束数据集大小字典。
    class_names = image_datasets["train"].classes  # EN: Get class names from folder names. CN: 从文件夹名称自动获取类别名。
    print("Dataset loaded successfully.")  # EN: Confirm successful loading. CN: 打印数据集加载成功信息。
    print(f"Classes: {class_names}")  # EN: Print class names, e.g., ants and bees. CN: 打印类别名，如 ants 和 bees。
    print(f"Training images: {dataset_sizes['train']}")  # EN: Print number of training images. CN: 打印训练图像数量。
    print(f"Validation images: {dataset_sizes['val']}")  # EN: Print number of validation images. CN: 打印验证图像数量。
    return dataloaders, dataset_sizes, class_names, data_transforms  # EN: Return all data-related objects. CN: 返回所有数据相关对象。


# ============================================================
# 4. Visualization helpers
# 4. 可视化辅助函数
# ============================================================

def unnormalize_image(tensor_img):  # EN: Define function to undo ImageNet normalization. CN: 定义反归一化函数。
    img = tensor_img.cpu().numpy().transpose((1, 2, 0))  # EN: Convert CHW tensor to HWC NumPy image. CN: 将 CHW Tensor 转为 HWC NumPy 图像。
    mean = np.array([0.485, 0.456, 0.406])  # EN: ImageNet mean. CN: ImageNet 均值。
    std = np.array([0.229, 0.224, 0.225])  # EN: ImageNet std. CN: ImageNet 标准差。
    img = std * img + mean  # EN: Reverse normalization. CN: 反归一化。
    img = np.clip(img, 0, 1)  # EN: Clip values to valid display range. CN: 将像素值限制到 [0,1]。
    return img  # EN: Return displayable image. CN: 返回可以显示的图像。


def show_training_batch(dataloaders, class_names):  # EN: Show one batch of training images. CN: 显示一批训练图像。
    inputs, labels = next(iter(dataloaders["train"]))  # EN: Get one batch from training loader. CN: 从训练集 DataLoader 取一个 batch。
    grid = make_grid(inputs[:min(8, len(inputs))])  # EN: Create image grid from up to 8 images. CN: 将最多 8 张图像组成网格。
    plt.figure(figsize=(8, 6))  # EN: Create a figure. CN: 创建画布。
    plt.imshow(unnormalize_image(grid))  # EN: Show unnormalized grid. CN: 显示反归一化后的图像网格。
    plt.title([class_names[i] for i in labels[:min(8, len(labels))]])  # EN: Show class names as title. CN: 将类别名显示在标题中。
    plt.axis("off")  # EN: Hide axes. CN: 隐藏坐标轴。
    plt.show()  # EN: Display figure. CN: 显示图像。


# ============================================================
# 5. Mini Vision Transformer model
# 5. 最小 Vision Transformer 模型
# ============================================================

class PatchEmbedding(nn.Module):  # EN: Define patch embedding module. CN: 定义 patch embedding 模块。
    def __init__(self, img_size, patch_size, in_channels, d_model):  # EN: Initialize patch embedding. CN: 初始化 patch embedding。
        super().__init__()  # EN: Initialize parent nn.Module. CN: 初始化父类 nn.Module。
        self.img_size = img_size  # EN: Store input image size. CN: 保存输入图像大小。
        self.patch_size = patch_size  # EN: Store patch size. CN: 保存 patch 大小。
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)  
        # EN: Calculate total number of patches. CN: 计算 patch 总数。
        self.proj = nn.Conv2d(  
            # EN: Use Conv2d to split image into patches and project features. CN: 使用 Conv2d 同时完成切 patch 和特征投影。
            in_channels=in_channels,  
            # EN: Input channels, usually 3 for RGB. CN: 输入通道数，RGB 通常为 3。
            out_channels=d_model,  # EN: Output embedding dimension. CN: 输出 embedding 维度。
            kernel_size=patch_size,  # EN: Kernel size equals patch size. CN: 卷积核大小等于 patch 大小。
            stride=patch_size  # EN: Stride equals patch size, so patches do not overlap. CN: 步长等于 patch 大小，因此 patch 不重叠。
        )  # EN: End Conv2d. CN: 结束 Conv2d。

    def forward(self, x):  # EN: Forward pass for patch embedding. CN: patch embedding 的前向传播。
        x = self.proj(x)  
        # EN: Convert image to patch feature map: [B, D, H/P, W/P]. CN: 将图像转为 patch 特征图：[B,D,H/P,W/P]。
        x = x.flatten(2)  
        # EN: Flatten spatial dimensions: [B, D, N]. CN: 展平空间维度：[B,D,N]。
        x = x.transpose(1, 2)  
        # EN: Convert to sequence format: [B, N, D]. CN: 转成序列格式：[B,N,D]。
        return x  # EN: Return patch token sequence. CN: 返回 patch token 序列。


class MiniViT(nn.Module):  # EN: Define a small Vision Transformer classifier. CN: 定义一个小型 Vision Transformer 分类器。
    def __init__(self, img_size, patch_size, in_channels, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout):  # EN: Initialize MiniViT. CN: 初始化 MiniViT。
        super().__init__()  # EN: Initialize parent nn.Module. CN: 初始化父类 nn.Module。
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)  # EN: Create patch embedding layer. CN: 创建 patch embedding 层。
        num_patches = self.patch_embed.num_patches  # EN: Get number of image patches. CN: 获取图像 patch 数量。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # EN: Learnable class token. CN: 可学习的分类 token。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))  # EN: Learnable positional embedding. CN: 可学习的位置编码。
        self.dropout = nn.Dropout(dropout)  # EN: Dropout after adding positional embedding. CN: 加位置编码后的 dropout。
        encoder_layer = nn.TransformerEncoderLayer(  # EN: Create one Transformer encoder layer. CN: 创建一个 Transformer Encoder 层。
            d_model=d_model,  # EN: Token embedding dimension. CN: token embedding 维度。
            nhead=nhead,  # EN: Number of attention heads. CN: 注意力 head 数量。
            dim_feedforward=dim_feedforward,  # EN: Feed-forward hidden dimension. CN: 前馈网络隐藏维度。
            dropout=dropout,  # EN: Dropout inside Transformer. CN: Transformer 内部 dropout。
            activation="gelu",  # EN: GELU activation is common in Transformers. CN: Transformer 常用 GELU 激活函数。
            batch_first=True  # EN: Input shape is [B, sequence, feature]. CN: 输入格式为 [B,序列长度,特征维度]。
        )  # EN: End encoder layer. CN: 结束 Encoder 层定义。
        self.encoder = nn.TransformerEncoder(  # EN: Stack Transformer encoder layers. CN: 堆叠 Transformer Encoder 层。
            encoder_layer=encoder_layer,  # EN: Base encoder layer. CN: 基础 Encoder 层。
            num_layers=num_layers  # EN: Number of stacked layers. CN: 堆叠层数。
        )  # EN: End Transformer encoder. CN: 结束 Transformer Encoder。
        self.norm = nn.LayerNorm(d_model)  # EN: Layer normalization before classifier. CN: 分类头前的 LayerNorm。
        self.head = nn.Linear(d_model, num_classes)  # EN: Final classification layer. CN: 最终分类层。
        self._init_weights()  # EN: Initialize learnable parameters. CN: 初始化可学习参数。

    def _init_weights(self):  # EN: Define parameter initialization. CN: 定义参数初始化函数。
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # EN: Initialize positional embedding. CN: 初始化位置编码。
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # EN: Initialize class token. CN: 初始化分类 token。
        nn.init.trunc_normal_(self.head.weight, std=0.02)  # EN: Initialize classifier weight. CN: 初始化分类层权重。
        nn.init.zeros_(self.head.bias)  # EN: Initialize classifier bias as zero. CN: 将分类层 bias 初始化为 0。

    def forward(self, x):  # EN: Forward pass of MiniViT. CN: MiniViT 的前向传播。
        batch_size = x.shape[0]  # EN: Get batch size. CN: 获取 batch size。
        x = self.patch_embed(x)  # EN: Convert images to patch tokens: [B, N, D]. CN: 将图像转为 patch tokens：[B,N,D]。
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # EN: Copy class token for each image. CN: 为每张图像复制分类 token。
        x = torch.cat((cls_tokens, x), dim=1)  # EN: Add class token to patch sequence. CN: 将分类 token 拼接到 patch 序列前面。
        x = x + self.pos_embed  # EN: Add positional embedding. CN: 加上位置编码。
        x = self.dropout(x)  # EN: Apply dropout. CN: 应用 dropout。
        x = self.encoder(x)  # EN: Pass token sequence through Transformer encoder. CN: 将 token 序列输入 Transformer Encoder。
        cls_output = x[:, 0]  # EN: Use output of class token for classification. CN: 使用分类 token 的输出做分类。
        cls_output = self.norm(cls_output)  # EN: Normalize class token output. CN: 对分类 token 输出做归一化。
        logits = self.head(cls_output)  # EN: Produce class logits. CN: 输出类别 logits。
        return logits  # EN: Return raw logits; CrossEntropyLoss will apply softmax internally. CN: 返回原始 logits；CrossEntropyLoss 内部会处理 softmax。


# ============================================================
# 6. Build model
# 6. 构建模型
# ============================================================

def build_model(num_classes):  # EN: Build MiniViT model based on class number. CN: 根据类别数构建 MiniViT 模型。
    model = MiniViT(  # EN: Create MiniViT instance. CN: 创建 MiniViT 实例。
        img_size=IMG_SIZE,  # EN: Input image size. CN: 输入图像大小。
        patch_size=PATCH_SIZE,  # EN: Patch size. CN: patch 大小。
        in_channels=IN_CHANNELS,  # EN: Input channels. CN: 输入通道数。
        num_classes=num_classes,  # EN: Number of output classes. CN: 输出类别数。
        d_model=D_MODEL,  # EN: Embedding dimension. CN: embedding 维度。
        nhead=NHEAD,  # EN: Number of attention heads. CN: 注意力 head 数量。
        num_layers=NUM_LAYERS,  # EN: Number of encoder layers. CN: Encoder 层数。
        dim_feedforward=DIM_FEEDFORWARD,  # EN: Feed-forward hidden dimension. CN: 前馈网络隐藏维度。
        dropout=DROPOUT  # EN: Dropout rate. CN: dropout 比例。
    )  # EN: End model creation. CN: 结束模型创建。
    return model  # EN: Return model. CN: 返回模型。


# ============================================================
# 7. Training and validation
# 7. 训练和验证
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):  # EN: Train model for one epoch. CN: 训练模型一个 epoch。
    model.train()  # EN: Set model to training mode. CN: 将模型设置为训练模式。
    running_loss = 0.0  # EN: Accumulate training loss. CN: 累计训练损失。
    running_corrects = 0  # EN: Count correct predictions. CN: 统计正确预测数量。
    total_samples = 0  # EN: Count total samples. CN: 统计总样本数量。
    for inputs, labels in dataloader:  # EN: Iterate over mini-batches. CN: 遍历每个 mini-batch。
        inputs = inputs.to(device)  # EN: Move images to CPU/GPU device. CN: 将图像移动到 CPU/GPU。
        labels = labels.to(device)  # EN: Move labels to CPU/GPU device. CN: 将标签移动到 CPU/GPU。
        optimizer.zero_grad()  # EN: Clear previous gradients. CN: 清空上一轮梯度。
        outputs = model(inputs)  # EN: Forward pass to get logits. CN: 前向传播得到 logits。
        loss = criterion(outputs, labels)  # EN: Compute classification loss. CN: 计算分类损失。
        _, preds = torch.max(outputs, dim=1)  # EN: Get predicted class indices. CN: 获取预测类别索引。
        loss.backward()  # EN: Backpropagate gradients. CN: 反向传播计算梯度。
        optimizer.step()  # EN: Update model parameters. CN: 更新模型参数。
        running_loss += loss.item() * inputs.size(0)  
        # EN: Accumulate loss weighted by batch size. CN: 按 batch size 累计 loss。
        running_corrects += torch.sum(preds == labels.data).item()  # EN: Count correct predictions. CN: 累计正确预测数量。
        total_samples += inputs.size(0)  # EN: Update sample count. CN: 更新样本总数。
    epoch_loss = running_loss / total_samples  # EN: Compute average loss. CN: 计算平均 loss。
    epoch_acc = running_corrects / total_samples  # EN: Compute accuracy. CN: 计算准确率。
    return epoch_loss, epoch_acc  # EN: Return training loss and accuracy. CN: 返回训练 loss 和准确率。


def evaluate(model, dataloader, criterion, device):  # EN: Evaluate model on validation set. CN: 在验证集上评估模型。
    model.eval()  # EN: Set model to evaluation mode. CN: 将模型设置为评估模式。
    running_loss = 0.0  # EN: Accumulate validation loss. CN: 累计验证损失。
    running_corrects = 0  # EN: Count correct predictions. CN: 统计正确预测数量。
    total_samples = 0  # EN: Count total samples. CN: 统计总样本数量。
    with torch.no_grad():  # EN: Disable gradient computation for validation. CN: 验证时不计算梯度。
        for inputs, labels in dataloader:  # EN: Iterate over validation batches. CN: 遍历验证集 batch。
            inputs = inputs.to(device)  # EN: Move images to device. CN: 将图像移动到设备。
            labels = labels.to(device)  # EN: Move labels to device. CN: 将标签移动到设备。
            outputs = model(inputs)  # EN: Forward pass. CN: 前向传播。
            loss = criterion(outputs, labels)  # EN: Compute validation loss. CN: 计算验证 loss。
            _, preds = torch.max(outputs, dim=1)  # EN: Get predicted class indices. CN: 获取预测类别索引。
            running_loss += loss.item() * inputs.size(0)  # EN: Accumulate validation loss. CN: 累计验证 loss。
            running_corrects += torch.sum(preds == labels.data).item()  # EN: Count correct predictions. CN: 累计正确预测数量。
            total_samples += inputs.size(0)  # EN: Update sample count. CN: 更新样本数量。
    epoch_loss = running_loss / total_samples  # EN: Compute average validation loss. CN: 计算平均验证 loss。
    epoch_acc = running_corrects / total_samples  # EN: Compute validation accuracy. CN: 计算验证准确率。
    return epoch_loss, epoch_acc  # EN: Return validation loss and accuracy. CN: 返回验证 loss 和准确率。


def train_model(model, dataloaders, dataset_sizes, class_names, device):  # EN: Full training pipeline. CN: 完整训练流程。
    criterion = nn.CrossEntropyLoss()  # EN: Standard loss for multi-class classification. CN: 多类别分类的标准损失函数。
    optimizer = optim.AdamW(  # EN: AdamW optimizer is commonly used for Transformers. CN: Transformer 常用 AdamW 优化器。
        model.parameters(),  # EN: Optimize all model parameters. CN: 优化所有模型参数。
        lr=LEARNING_RATE,  # EN: Set learning rate. CN: 设置学习率。
        weight_decay=WEIGHT_DECAY  # EN: Set weight decay. CN: 设置权重衰减。
    )  # EN: End optimizer definition. CN: 结束优化器定义。
    scheduler = lr_scheduler.StepLR(  # EN: Step learning-rate scheduler. CN: 阶梯式学习率调度器。
        optimizer,  # EN: Optimizer to schedule. CN: 需要调度的优化器。
        step_size=STEP_SIZE,  # EN: Decay every STEP_SIZE epochs. CN: 每 STEP_SIZE 个 epoch 衰减一次。
        gamma=GAMMA  # EN: Decay factor. CN: 衰减系数。
    )  # EN: End scheduler definition. CN: 结束调度器定义。
    best_model_wts = copy.deepcopy(model.state_dict())  # EN: Copy initial model weights. CN: 复制初始模型权重。
    best_acc = 0.0  # EN: Track best validation accuracy. CN: 记录最佳验证准确率。
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}  
    # EN: Store training curves. CN: 保存训练曲线数据。
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # EN: Create output folder if not exists. CN: 如果输出文件夹不存在则创建。
    since = time.time()  # EN: Record starting time. CN: 记录开始时间。
    for epoch in range(NUM_EPOCHS):  # EN: Loop over epochs. CN: 遍历每个 epoch。
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")  # EN: Print current epoch. CN: 打印当前 epoch。
        print("-" * 30)  # EN: Print separator. CN: 打印分隔线。
        train_loss, train_acc = train_one_epoch(  # EN: Train for one epoch. CN: 训练一个 epoch。
            model=model,  # EN: Model to train. CN: 要训练的模型。
            dataloader=dataloaders["train"],  # EN: Training dataloader. CN: 训练集 DataLoader。
            criterion=criterion,  # EN: Loss function. CN: 损失函数。
            optimizer=optimizer,  # EN: Optimizer. CN: 优化器。
            device=device  # EN: CPU/GPU device. CN: CPU/GPU 设备。
        )  # EN: End train_one_epoch call. CN: 结束 train_one_epoch 调用。
        val_loss, val_acc = evaluate(  # EN: Evaluate after each epoch. CN: 每个 epoch 后进行验证。
            model=model,  # EN: Model to evaluate. CN: 要验证的模型。
            dataloader=dataloaders["val"],  # EN: Validation dataloader. CN: 验证集 DataLoader。
            criterion=criterion,  # EN: Loss function. CN: 损失函数。
            device=device  # EN: CPU/GPU device. CN: CPU/GPU 设备。
        )  # EN: End evaluation call. CN: 结束验证调用。
        scheduler.step()  # EN: Update learning rate. CN: 更新学习率。
        history["train_loss"].append(train_loss)  # EN: Save train loss. CN: 保存训练 loss。
        history["train_acc"].append(train_acc)  # EN: Save train accuracy. CN: 保存训练准确率。
        history["val_loss"].append(val_loss)  # EN: Save validation loss. CN: 保存验证 loss。
        history["val_acc"].append(val_acc)  # EN: Save validation accuracy. CN: 保存验证准确率。
        print(f"train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")  # EN: Print train results. CN: 打印训练结果。
        print(f"val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")  # EN: Print validation results. CN: 打印验证结果。
        if val_acc > best_acc:  # EN: If current model is better. CN: 如果当前模型更好。
            best_acc = val_acc  # EN: Update best accuracy. CN: 更新最佳准确率。
            best_model_wts = copy.deepcopy(model.state_dict())  # EN: Save best weights in memory. CN: 在内存中保存最佳权重。
            torch.save(  # EN: Save checkpoint to disk. CN: 将 checkpoint 保存到硬盘。
                {  # EN: Create checkpoint dictionary. CN: 创建 checkpoint 字典。
                    "model_state_dict": best_model_wts,  # EN: Save model weights. CN: 保存模型权重。
                    "class_names": class_names,  # EN: Save class names. CN: 保存类别名。
                    "num_classes": len(class_names),  # EN: Save class number. CN: 保存类别数量。
                    "img_size": IMG_SIZE,  # EN: Save image size. CN: 保存图像大小。
                    "patch_size": PATCH_SIZE,  # EN: Save patch size. CN: 保存 patch 大小。
                    "d_model": D_MODEL,  # EN: Save embedding dimension. CN: 保存 embedding 维度。
                    "history": history,  # EN: Save training history. CN: 保存训练历史。
                    "best_val_acc": best_acc  # EN: Save best validation accuracy. CN: 保存最佳验证准确率。
                },  # EN: End checkpoint dictionary. CN: 结束 checkpoint 字典。
                CHECKPOINT_PATH  # EN: Checkpoint output path. CN: checkpoint 输出路径。
            )  # EN: End torch.save. CN: 结束 torch.save。
            print(f"Saved best checkpoint to: {CHECKPOINT_PATH}")  # EN: Print checkpoint path. CN: 打印 checkpoint 路径。
    elapsed = time.time() - since  # EN: Calculate elapsed time. CN: 计算训练耗时。
    print("\nTraining complete.")  # EN: Print training complete. CN: 打印训练完成。
    print(f"Training time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s")  # EN: Print training time. CN: 打印训练时间。
    print(f"Best validation accuracy: {best_acc:.4f}")  # EN: Print best validation accuracy. CN: 打印最佳验证准确率。
    model.load_state_dict(best_model_wts)  # EN: Load best weights back into model. CN: 将最佳权重加载回模型。
    return model, history  # EN: Return best model and history. CN: 返回最佳模型和训练历史。


# ============================================================
# 8. Plot curves and visualize predictions
# 8. 绘制曲线和可视化预测
# ============================================================

def plot_history(history):  # EN: Plot loss and accuracy curves. CN: 绘制 loss 和 accuracy 曲线。
    epochs = range(1, len(history["train_loss"]) + 1)  # EN: Create epoch index. CN: 创建 epoch 序号。
    plt.figure(figsize=(8, 6))  # EN: Create figure for loss. CN: 创建 loss 曲线画布。
    plt.plot(epochs, history["train_loss"], label="Train Loss")  # EN: Plot train loss. CN: 绘制训练 loss。
    plt.plot(epochs, history["val_loss"], label="Val Loss")  # EN: Plot validation loss. CN: 绘制验证 loss。
    plt.xlabel("Epoch")  # EN: X-axis label. CN: X 轴标签。
    plt.ylabel("Loss")  # EN: Y-axis label. CN: Y 轴标签。
    plt.title("Mini ViT Loss Curve")  # EN: Figure title. CN: 图标题。
    plt.legend()  # EN: Show legend. CN: 显示图例。
    plt.grid(True)  # EN: Show grid. CN: 显示网格。
    loss_path = HISTORY_PLOT_PATH.replace(".png", "_loss.png")  # EN: Build loss plot path. CN: 构建 loss 曲线保存路径。
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")  # EN: Save loss plot. CN: 保存 loss 曲线。
    plt.show()  # EN: Display loss plot. CN: 显示 loss 曲线。
    plt.figure(figsize=(8, 6))  # EN: Create figure for accuracy. CN: 创建准确率曲线画布。
    plt.plot(epochs, history["train_acc"], label="Train Acc")  # EN: Plot train accuracy. CN: 绘制训练准确率。
    plt.plot(epochs, history["val_acc"], label="Val Acc")  # EN: Plot validation accuracy. CN: 绘制验证准确率。
    plt.xlabel("Epoch")  # EN: X-axis label. CN: X 轴标签。
    plt.ylabel("Accuracy")  # EN: Y-axis label. CN: Y 轴标签。
    plt.title("Mini ViT Accuracy Curve")  # EN: Figure title. CN: 图标题。
    plt.legend()  # EN: Show legend. CN: 显示图例。
    plt.grid(True)  # EN: Show grid. CN: 显示网格。
    acc_path = HISTORY_PLOT_PATH.replace(".png", "_acc.png")  # EN: Build accuracy plot path. CN: 构建准确率曲线保存路径。
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")  # EN: Save accuracy plot. CN: 保存准确率曲线。
    plt.show()  # EN: Display accuracy plot. CN: 显示准确率曲线。
    print(f"Saved plots to: {loss_path} and {acc_path}")  # EN: Print saved plot paths. CN: 打印曲线保存路径。


def visualize_model(model, dataloaders, class_names, device, num_images=6):  
    # EN: Visualize predictions on validation images. CN: 可视化验证集预测结果。
    model.eval()  # EN: Set model to evaluation mode. CN: 设置模型为评估模式。
    images_shown = 0  # EN: Count displayed images. CN: 统计已显示图像数量。
    plt.figure(figsize=(8, 8))  # EN: Create figure. CN: 创建画布。
    with torch.no_grad():  # EN: Disable gradients for visualization. CN: 可视化时不计算梯度。
        for inputs, labels in dataloaders["val"]:  
            # EN: Iterate over validation batches. CN: 遍历验证集 batch。
            inputs = inputs.to(device)  # EN: Move images to device. CN: 将图像移动到设备。
            labels = labels.to(device)  # EN: Move labels to device. CN: 将标签移动到设备。
            outputs = model(inputs)  # EN: Run model prediction. CN: 执行模型预测。
            _, preds = torch.max(outputs, dim=1)  # EN: Get predicted class indices. CN: 获取预测类别索引。
            for j in range(inputs.size(0)):  # EN: Loop through images in batch. CN: 遍历 batch 内图像。
                images_shown += 1  # EN: Increase displayed image count. CN: 显示数量加一。
                ax = plt.subplot(num_images // 2, 2, images_shown)  # EN: Create subplot. CN: 创建子图。
                ax.axis("off")  # EN: Hide axes. CN: 隐藏坐标轴。
                pred_name = class_names[preds[j].item()]  
                # EN: Convert prediction index to class name. CN: 将预测索引转换成类别名。
                true_name = class_names[labels[j].item()]  
                # EN: Convert true label index to class name. CN: 将真实标签索引转换成类别名。
                ax.set_title(f"Pred: {pred_name}\nTrue: {true_name}")  # EN: Show prediction and ground truth. CN: 显示预测类别和真实类别。
                ax.imshow(unnormalize_image(inputs.cpu()[j]))  # EN: Show image. CN: 显示图像。
                if images_shown == num_images:  
                    # EN: Stop after enough images. CN: 显示足够图像后停止。
                    plt.show()  # EN: Display figure. CN: 显示图像。
                    return  # EN: Exit function. CN: 退出函数。
    plt.show()  # EN: Display figure if fewer images. CN: 如果图像不足，也显示画布。


# ============================================================
# 9. Single image inference
# 9. 单张图片预测
# ============================================================

def predict_single_image(model, img_path, data_transforms, class_names, device):  # EN: Predict one image. CN: 对单张图片进行预测。
    if not os.path.exists(img_path):  # EN: Check if image exists. CN: 检查图片是否存在。
        print(f"Image does not exist: {img_path}")  # EN: Print error message. CN: 打印错误信息。
        return  # EN: Stop prediction. CN: 停止预测。
    model.eval()  # EN: Set model to evaluation mode. CN: 设置模型为评估模式。
    img = Image.open(img_path).convert("RGB")  # EN: Open image and convert to RGB. CN: 打开图片并转换为 RGB。
    img_tensor = data_transforms["val"](img)  # EN: Apply validation transform. CN: 应用验证集预处理。
    img_tensor = img_tensor.unsqueeze(0).to(device)  # EN: Add batch dimension and move to device. CN: 增加 batch 维度并移动到设备。
    with torch.no_grad():  # EN: Disable gradient computation. CN: 不计算梯度。
        logits = model(img_tensor)  # EN: Get raw logits. CN: 得到原始 logits。
        probs = torch.softmax(logits, dim=1)  # EN: Convert logits to probabilities. CN: 将 logits 转换成概率。
        conf, pred = torch.max(probs, dim=1)  # EN: Get confidence and predicted class. CN: 获取置信度和预测类别。
    pred_name = class_names[pred.item()]  
    # EN: Convert class index to class name. CN: 将类别索引转换成类别名。
    confidence = conf.item()  
    # EN: Convert confidence to Python float. CN: 将置信度转换成 Python 浮点数。
    print(f"Image: {img_path}")  # EN: Print image path. CN: 打印图片路径。
    print(f"Predicted class: {pred_name}")  # EN: Print predicted class. CN: 打印预测类别。
    print(f"Confidence: {confidence:.4f}")  # EN: Print confidence. CN: 打印置信度。
    plt.figure(figsize=(5, 5))  # EN: Create figure. CN: 创建画布。
    plt.imshow(img)  # EN: Show original image. CN: 显示原始图像。
    plt.axis("off")  # EN: Hide axes. CN: 隐藏坐标轴。
    plt.title(f"Predicted: {pred_name} ({confidence:.2%})")  # EN: Show prediction title. CN: 显示预测结果标题。
    plt.show()  # EN: Display image. CN: 显示图像。


# ============================================================
# 10. Main function
# 10. 主函数
# ============================================================

def main():  # EN: Main entry point. CN: 主程序入口。
    set_seed(SEED)  # EN: Set random seed. CN: 设置随机种子。
    device = get_device()  # EN: Select CPU or GPU. CN: 选择 CPU 或 GPU。
    dataloaders, dataset_sizes, class_names, data_transforms = build_dataloaders(  # EN: Build data pipeline. CN: 构建数据管道。
        data_dir=DATA_DIR,  # EN: Dataset root directory. CN: 数据集根目录。
        batch_size=BATCH_SIZE,  # EN: Batch size. CN: batch size。
        num_workers=NUM_WORKERS  # EN: Number of workers. CN: worker 数量。
    )  # EN: End dataloader building. CN: 结束 DataLoader 构建。
    show_training_batch(dataloaders, class_names)  
    # EN: Visualize one training batch. CN: 可视化一个训练 batch。
    num_classes = len(class_names)  
    # EN: Number of classes inferred from folders. CN: 根据文件夹自动得到类别数。
    model = build_model(num_classes=num_classes)  
    # EN: Build MiniViT model. CN: 构建 MiniViT 模型。
    model = model.to(device)  
    # EN: Move model to CPU/GPU. CN: 将模型移动到 CPU/GPU。
    print(model)  # EN: Print model structure. CN: 打印模型结构。
    model, history = train_model(  # EN: Train model and get history. CN: 训练模型并获得训练历史。
        model=model,  # EN: Model to train. CN: 要训练的模型。
        dataloaders=dataloaders,  # EN: Dataloaders. CN: DataLoader 字典。
        dataset_sizes=dataset_sizes,  # EN: Dataset sizes. CN: 数据集大小。
        class_names=class_names,  # EN: Class names. CN: 类别名称。
        device=device  # EN: CPU/GPU device. CN: CPU/GPU 设备。
    )  # EN: End training call. CN: 结束训练调用。
    plot_history(history)  # EN: Plot loss and accuracy curves. CN: 绘制 loss 和 accuracy 曲线。
    visualize_model(  # EN: Show predictions on validation images. CN: 显示验证图像预测结果。
        model=model,  # EN: Trained model. CN: 训练好的模型。
        dataloaders=dataloaders,  # EN: Dataloaders. CN: DataLoader 字典。
        class_names=class_names,  # EN: Class names. CN: 类别名称。
        device=device,  # EN: CPU/GPU device. CN: CPU/GPU 设备。
        num_images=6  # EN: Number of validation images to show. CN: 要显示的验证图像数量。
    )  # EN: End visualization. CN: 结束可视化。
    test_img_path = os.path.join(DATA_DIR, "val", class_names[0], os.listdir(os.path.join(DATA_DIR, "val", class_names[0]))[0])  
    # EN: Pick one validation image automatically. CN: 自动选择一张验证图片。
    predict_single_image(  # EN: Run single-image prediction. CN: 执行单张图片预测。
        model=model,  # EN: Trained model. CN: 训练好的模型。
        img_path=test_img_path,  # EN: Test image path. CN: 测试图片路径。
        data_transforms=data_transforms,  # EN: Transform dictionary. CN: 预处理字典。
        class_names=class_names,  # EN: Class names. CN: 类别名称。
        device=device  # EN: CPU/GPU device. CN: CPU/GPU 设备。
    )  # EN: End single-image prediction. CN: 结束单张图片预测。


if __name__ == "__main__":  # EN: Run main only when this file is executed directly. CN: 只有直接运行本文件时才执行 main。
    main()  # EN: Start the full pipeline. CN: 启动完整流程。
