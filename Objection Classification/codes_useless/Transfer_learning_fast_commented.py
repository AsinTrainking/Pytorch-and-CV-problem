# ============================================================
# Transfer Learning Fast Version: Fixed Feature Extractor
# 迁移学习快速版本：固定特征提取器
# ============================================================

# Import PyTorch core library.
# 导入 PyTorch 核心库。
import torch

# Import torchvision, which provides datasets, models, and image transforms.
# 导入 torchvision，用于加载数据集、预训练模型和图像预处理方法。
import torchvision

# Import neural network modules from PyTorch.
# 导入 PyTorch 的神经网络模块。
import torch.nn as nn

# Import optimization algorithms such as SGD and Adam.
# 导入优化器模块，例如 SGD 和 Adam。
import torch.optim as optim

# Import learning rate scheduler.
# 导入学习率调度器，用于训练过程中调整 learning rate。
from torch.optim import lr_scheduler

# Import cuDNN backend control.
# 导入 cuDNN 后端控制模块，用于加速 GPU 卷积计算。
import torch.backends.cudnn as cudnn

# Import NumPy for numerical operations.
# 导入 NumPy，用于数值计算。
import numpy as np

# Import dataset loader, pretrained models, and image transforms.
# 导入数据集读取工具、预训练模型和图像变换工具。
from torchvision import datasets, models, transforms

# Import matplotlib for visualizing images and training results.
# 导入 matplotlib，用于图像和结果可视化。
import matplotlib.pyplot as plt

# Import time module to record training time.
# 导入 time 模块，用于统计训练时间。
import time

# Import os module for file path operations.
# 导入 os 模块，用于处理文件路径。
import os

# Import PIL Image for loading a single custom image during inference.
# 导入 PIL Image，用于读取单张测试图片。
from PIL import Image

# Import TemporaryDirectory for temporary checkpoint storage during training.
# 导入 TemporaryDirectory，用于训练过程中临时保存最佳模型参数。
from tempfile import TemporaryDirectory


# ============================================================
# 1. Basic settings
# 1. 基本设置
# ============================================================

# Enable cuDNN auto-tuner to find the best convolution algorithm for fixed input sizes.
# 开启 cuDNN 自动优化，对于固定输入尺寸的卷积网络可以加速训练。
cudnn.benchmark = True

# Turn on matplotlib interactive mode.
# 打开 matplotlib 交互模式，使图像可以动态显示。
plt.ion()

# Define the dataset root directory.
# 定义数据集根目录。
data_dir = 'hymenoptera_data'

# Define batch size.
# 定义 batch size，每次送入模型的图像数量。
batch_size = 8

# Define number of training epochs.
# 定义训练轮数。
num_epochs = 25

# Set num_workers to 0 for Windows compatibility.
# 在 Windows 系统中建议设置为 0，避免 DataLoader 多进程报错。
num_workers = 0

# Automatically select GPU if CUDA is available; otherwise use CPU.
# 如果 CUDA 可用则自动使用 GPU，否则使用 CPU。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Print the selected device.
# 打印当前使用的计算设备。
print(f"Using {device} device")


# ============================================================
# 2. Data transforms
# 2. 数据预处理和数据增强
# ============================================================

# Define image preprocessing pipelines for training and validation.
# 定义训练集和验证集的图像预处理流程。
data_transforms = {

    # Training data transform.
    # 训练集的数据增强和归一化。
    'train': transforms.Compose([

        # Randomly crop and resize each image to 224x224.
        # 随机裁剪并缩放图像到 224x224，用于增强数据多样性。
        transforms.RandomResizedCrop(224),

        # Randomly flip image horizontally.
        # 随机水平翻转图像，用于增强模型泛化能力。
        transforms.RandomHorizontalFlip(),

        # Convert PIL image or NumPy array to PyTorch tensor.
        # 将 PIL 图像或 NumPy 数组转换为 PyTorch Tensor。
        transforms.ToTensor(),

        # Normalize image using ImageNet mean and standard deviation.
        # 使用 ImageNet 的均值和标准差进行归一化，因为 ResNet18 是在 ImageNet 上预训练的。
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    # Validation data transform.
    # 验证集的数据预处理，不使用随机增强，保证评估稳定。
    'val': transforms.Compose([

        # Resize the shorter side of the image to 256.
        # 将图像较短边缩放到 256。
        transforms.Resize(256),

        # Center crop the image to 224x224.
        # 从图像中心裁剪出 224x224 的区域。
        transforms.CenterCrop(224),

        # Convert image to PyTorch tensor.
        # 将图像转换为 PyTorch Tensor。
        transforms.ToTensor(),

        # Normalize using ImageNet statistics.
        # 使用 ImageNet 的均值和标准差进行归一化。
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
}


# ============================================================
# 3. Dataset and DataLoader
# 3. 数据集读取和 DataLoader 构建
# ============================================================

# Create ImageFolder datasets for train and validation folders.
# 使用 ImageFolder 读取 train 和 val 文件夹中的图像数据。
image_datasets = {

    # x will be either 'train' or 'val'.
    # x 代表当前读取的数据集类型，可能是 'train' 或 'val'。
    x: datasets.ImageFolder(

        # Build the path such as hymenoptera_data/train or hymenoptera_data/val.
        # 拼接路径，例如 hymenoptera_data/train 或 hymenoptera_data/val。
        os.path.join(data_dir, x),

        # Apply the corresponding transform.
        # 应用对应的图像预处理流程。
        data_transforms[x]
    )

    # Loop over training and validation phases.
    # 遍历训练集和验证集。
    for x in ['train', 'val']
}

# Create DataLoader for train and validation datasets.
# 为训练集和验证集创建 DataLoader。
dataloaders = {

    # x will be 'train' or 'val'.
    # x 代表当前阶段，是 'train' 或 'val'。
    x: torch.utils.data.DataLoader(

        # Use the corresponding dataset.
        # 使用对应的数据集。
        image_datasets[x],

        # Number of images in each batch.
        # 每个 batch 中的图像数量。
        batch_size=batch_size,

        # Shuffle data. For simplicity, here both train and val are shuffled.
        # 是否打乱数据。这里为了简单，train 和 val 都设置为 True。
        shuffle=True,

        # Number of subprocesses for data loading.
        # 数据加载使用的子进程数量；Windows 下建议为 0。
        num_workers=num_workers
    )

    # Build DataLoader for both train and val.
    # 为 train 和 val 分别构建 DataLoader。
    for x in ['train', 'val']
}

# Store the number of images in each dataset.
# 保存训练集和验证集中的图像数量。
dataset_sizes = {

    # Get the length of each dataset.
    # 获取每个数据集的样本数量。
    x: len(image_datasets[x])

    # Loop over train and val.
    # 遍历 train 和 val。
    for x in ['train', 'val']
}

# Get class names from the training folder names.
# 从训练集文件夹名称中获取类别名称，例如 ['ants', 'bees']。
class_names = image_datasets['train'].classes

# Print class names.
# 打印类别名称。
print("Class names:", class_names)

# Print dataset sizes.
# 打印训练集和验证集样本数量。
print("Dataset sizes:", dataset_sizes)


# ============================================================
# 4. Image visualization helper
# 4. 图像可视化辅助函数
# ============================================================

# Define a function to show a tensor image.
# 定义一个函数，用于显示 Tensor 格式的图像。
def imshow(inp, title=None):

    # Convert tensor from shape [C, H, W] to [H, W, C].
    # 将 Tensor 形状从 [通道, 高, 宽] 转换为 [高, 宽, 通道]。
    inp = inp.numpy().transpose((1, 2, 0))

    # Define ImageNet mean values.
    # 定义 ImageNet 的均值。
    mean = np.array([0.485, 0.456, 0.406])

    # Define ImageNet standard deviation values.
    # 定义 ImageNet 的标准差。
    std = np.array([0.229, 0.224, 0.225])

    # Reverse normalization to recover displayable image.
    # 反归一化，将图像恢复到可显示的范围。
    inp = std * inp + mean

    # Clip pixel values to [0, 1].
    # 将像素值限制在 [0, 1] 范围内。
    inp = np.clip(inp, 0, 1)

    # Display the image.
    # 显示图像。
    plt.imshow(inp)

    # If a title is provided, show it.
    # 如果提供了标题，则显示标题。
    if title is not None:
        plt.title(title)

    # Pause briefly so that the plot can update.
    # 暂停一小段时间，让图像窗口更新。
    plt.pause(0.001)


# Get one batch of training data.
# 从训练集中取出一个 batch 的图像和标签。
inputs, classes = next(iter(dataloaders['train']))

# Make a grid image from the batch.
# 将一个 batch 中的多张图像拼接成网格图。
out = torchvision.utils.make_grid(inputs)

# Show the image grid with class labels.
# 显示图像网格，并把类别名称作为标题。
imshow(out, title=[class_names[x] for x in classes])


# ============================================================
# 5. Training function
# 5. 模型训练函数
# ============================================================

# Define a function to train and validate the model.
# 定义一个函数，用于训练和验证模型。
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    # Record the starting time.
    # 记录训练开始时间。
    since = time.time()

    # Create a temporary directory for saving the best model during training.
    # 创建一个临时文件夹，用于训练过程中保存最佳模型参数。
    with TemporaryDirectory() as tempdir:

        # Define temporary checkpoint path.
        # 定义临时 checkpoint 文件路径。
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        # Save the initial model parameters.
        # 保存初始模型参数。
        torch.save(model.state_dict(), best_model_params_path)

        # Initialize the best validation accuracy.
        # 初始化最佳验证准确率。
        best_acc = 0.0

        # Loop over all epochs.
        # 遍历所有训练轮数。
        for epoch in range(num_epochs):

            # Print current epoch.
            # 打印当前 epoch。
            print(f'Epoch {epoch}/{num_epochs - 1}')

            # Print separator line.
            # 打印分隔线。
            print('-' * 10)

            # Each epoch contains a training phase and a validation phase.
            # 每个 epoch 包含训练阶段和验证阶段。
            for phase in ['train', 'val']:

                # If current phase is training.
                # 如果当前阶段是训练阶段。
                if phase == 'train':

                    # Set model to training mode.
                    # 将模型设置为训练模式，启用 dropout/batch norm 的训练行为。
                    model.train()

                # If current phase is validation.
                # 如果当前阶段是验证阶段。
                else:

                    # Set model to evaluation mode.
                    # 将模型设置为评估模式，关闭 dropout，并固定 batch norm 行为。
                    model.eval()

                # Initialize accumulated loss.
                # 初始化累计 loss。
                running_loss = 0.0

                # Initialize accumulated correct predictions.
                # 初始化预测正确的样本数量。
                running_corrects = 0

                # Loop over all batches in the current phase.
                # 遍历当前阶段中的所有 batch。
                for inputs, labels in dataloaders[phase]:

                    # Move input images to GPU or CPU.
                    # 将输入图像移动到 GPU 或 CPU。
                    inputs = inputs.to(device)

                    # Move labels to GPU or CPU.
                    # 将标签移动到 GPU 或 CPU。
                    labels = labels.to(device)

                    # Clear previous gradients.
                    # 清空上一轮反向传播留下的梯度。
                    optimizer.zero_grad()

                    # Enable gradient computation only during training.
                    # 只有训练阶段才启用梯度计算，验证阶段不计算梯度以节省显存和时间。
                    with torch.set_grad_enabled(phase == 'train'):

                        # Forward pass: compute model outputs.
                        # 前向传播：计算模型输出。
                        outputs = model(inputs)

                        # Get predicted class index with maximum score.
                        # 取输出分数最大的类别作为预测结果。
                        _, preds = torch.max(outputs, 1)

                        # Compute classification loss.
                        # 计算分类损失。
                        loss = criterion(outputs, labels)

                        # If training phase, update model parameters.
                        # 如果是训练阶段，则进行反向传播和参数更新。
                        if phase == 'train':

                            # Backward pass: compute gradients.
                            # 反向传播：计算梯度。
                            loss.backward()

                            # Update trainable parameters.
                            # 使用优化器更新可训练参数。
                            optimizer.step()

                    # Accumulate batch loss multiplied by batch size.
                    # 累加当前 batch 的 loss，并乘以 batch size。
                    running_loss += loss.item() * inputs.size(0)

                    # Accumulate number of correct predictions.
                    # 累加预测正确的样本数量。
                    running_corrects += torch.sum(preds == labels.data)

                # Update learning rate after each training epoch.
                # 每个训练 epoch 结束后更新学习率。
                if phase == 'train':
                    scheduler.step()

                # Compute average loss for the current phase.
                # 计算当前阶段的平均 loss。
                epoch_loss = running_loss / dataset_sizes[phase]

                # Compute accuracy for the current phase.
                # 计算当前阶段的准确率。
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # Print loss and accuracy.
                # 打印 loss 和 accuracy。
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # If validation accuracy improves, save the model.
                # 如果验证集准确率提升，则保存当前模型参数。
                if phase == 'val' and epoch_acc > best_acc:

                    # Update best validation accuracy.
                    # 更新最佳验证准确率。
                    best_acc = epoch_acc

                    # Save the best model parameters.
                    # 保存最佳模型参数。
                    torch.save(model.state_dict(), best_model_params_path)

            # Print a blank line after each epoch.
            # 每个 epoch 结束后打印空行。
            print()

        # Compute total training time.
        # 计算总训练时间。
        time_elapsed = time.time() - since

        # Print total training time.
        # 打印总训练时间。
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # Print best validation accuracy.
        # 打印最佳验证准确率。
        print(f'Best val Acc: {best_acc:4f}')

        # Load the best model parameters back into the model.
        # 将最佳模型参数重新加载回模型中。
        model.load_state_dict(torch.load(best_model_params_path, map_location=device))

    # Return the best model.
    # 返回最佳模型。
    return model


# ============================================================
# 6. Visualization of model predictions
# 6. 模型预测结果可视化
# ============================================================

# Define a function to visualize validation predictions.
# 定义一个函数，用于可视化验证集上的预测结果。
def visualize_model(model, num_images=6):

    # Save the current training/evaluation state.
    # 保存模型当前的训练/评估状态。
    was_training = model.training

    # Set model to evaluation mode.
    # 将模型设置为评估模式。
    model.eval()

    # Count how many images have been shown.
    # 记录已经显示了多少张图片。
    images_so_far = 0

    # Create a new figure.
    # 创建一个新的图像窗口。
    fig = plt.figure()

    # Disable gradient computation for inference.
    # 推理阶段不需要计算梯度。
    with torch.no_grad():

        # Loop over validation batches.
        # 遍历验证集中的 batch。
        for i, (inputs, labels) in enumerate(dataloaders['val']):

            # Move images to GPU or CPU.
            # 将图像移动到 GPU 或 CPU。
            inputs = inputs.to(device)

            # Move labels to GPU or CPU.
            # 将标签移动到 GPU 或 CPU。
            labels = labels.to(device)

            # Forward pass.
            # 前向传播，得到模型输出。
            outputs = model(inputs)

            # Get predicted class indices.
            # 获取预测类别编号。
            _, preds = torch.max(outputs, 1)

            # Loop over images in the current batch.
            # 遍历当前 batch 中的每张图像。
            for j in range(inputs.size()[0]):

                # Increase displayed image count.
                # 已显示图像数量加 1。
                images_so_far += 1

                # Create subplot location.
                # 创建子图位置。
                ax = plt.subplot(num_images // 2, 2, images_so_far)

                # Hide axis.
                # 隐藏坐标轴。
                ax.axis('off')

                # Set subplot title as predicted class name.
                # 设置子图标题为预测类别名称。
                ax.set_title(f'predicted: {class_names[preds[j]]}')

                # Show image after moving it back to CPU.
                # 将图像移动回 CPU 并显示。
                imshow(inputs.cpu().data[j])

                # Stop when enough images have been shown.
                # 当显示足够数量的图像后停止。
                if images_so_far == num_images:

                    # Restore original model state.
                    # 恢复模型原来的训练/评估状态。
                    model.train(mode=was_training)

                    # Return from the function.
                    # 结束函数。
                    return

        # Restore original model state.
        # 恢复模型原来的训练/评估状态。
        model.train(mode=was_training)


# ============================================================
# 7. Build fixed feature extractor model
# 7. 构建固定特征提取器模型
# ============================================================

# Load a ResNet18 model pretrained on ImageNet.
# 加载在 ImageNet 数据集上预训练好的 ResNet18 模型。
model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')

# Freeze all pretrained parameters.
# 冻结所有预训练参数。
for param in model_conv.parameters():

    # Disable gradient computation for this parameter.
    # 不再为该参数计算梯度，因此训练时不会更新它。
    param.requires_grad = False

# Get the number of input features of the original final fully connected layer.
# 获取 ResNet18 原始最后一层全连接层的输入特征维度。
num_ftrs = model_conv.fc.in_features

# Replace the final fully connected layer with a new classifier.
# 替换最后一层全连接层，使其输出类别数等于当前任务类别数。
model_conv.fc = nn.Linear(num_ftrs, len(class_names))

# Move the model to GPU or CPU.
# 将模型移动到 GPU 或 CPU。
model_conv = model_conv.to(device)


# ============================================================
# 8. Loss function, optimizer, and scheduler
# 8. 损失函数、优化器和学习率调度器
# ============================================================

# Define cross-entropy loss for classification.
# 定义交叉熵损失函数，用于分类任务。
criterion = nn.CrossEntropyLoss()

# Define SGD optimizer.
# 定义 SGD 优化器。
optimizer_conv = optim.SGD(

    # Only optimize the final fully connected layer parameters.
    # 只优化最后一层全连接层的参数。
    model_conv.fc.parameters(),

    # Learning rate.
    # 学习率。
    lr=0.001,

    # Momentum helps accelerate SGD and stabilize optimization.
    # 动量项可以加速 SGD 收敛并提升优化稳定性。
    momentum=0.9
)

# Define step learning rate scheduler.
# 定义 StepLR 学习率调度器。
exp_lr_scheduler = lr_scheduler.StepLR(

    # The optimizer whose learning rate will be adjusted.
    # 需要被调整学习率的优化器。
    optimizer_conv,

    # Reduce learning rate every 7 epochs.
    # 每 7 个 epoch 降低一次学习率。
    step_size=7,

    # Multiply learning rate by 0.1 when scheduled.
    # 每次调整时将学习率乘以 0.1。
    gamma=0.1
)


# ============================================================
# 9. Train the model
# 9. 训练模型
# ============================================================

# Train and validate the fixed feature extractor model.
# 训练并验证固定特征提取器模型。
model_conv = train_model(

    # The model to train.
    # 要训练的模型。
    model_conv,

    # Loss function.
    # 损失函数。
    criterion,

    # Optimizer.
    # 优化器。
    optimizer_conv,

    # Learning rate scheduler.
    # 学习率调度器。
    exp_lr_scheduler,

    # Number of epochs.
    # 训练轮数。
    num_epochs=num_epochs
)

# Visualize predictions on validation images.
# 可视化模型在验证集图像上的预测结果。
visualize_model(model_conv)


# ============================================================
# 10. Save trained model permanently
# 10. 永久保存训练好的模型
# ============================================================

# Save the final trained model weights to a .pth file.
# 将最终训练好的模型权重永久保存为 .pth 文件。
torch.save(model_conv.state_dict(), 'resnet18_fixed_feature_extractor.pth')

# Print save message.
# 打印保存成功信息。
print("Model weights saved to resnet18_fixed_feature_extractor.pth")


# ============================================================
# 11. Single image inference function
# 11. 单张图像推理函数
# ============================================================

# Define a function to predict one custom image.
# 定义一个函数，用于预测单张自定义图像。
def visualize_model_predictions(model, img_path):

    # Save the current model state.
    # 保存模型当前状态。
    was_training = model.training

    # Set model to evaluation mode.
    # 将模型设置为评估模式。
    model.eval()

    # Open the image from file.
    # 从文件路径读取图像。
    img = Image.open(img_path).convert('RGB')

    # Apply validation transforms to the image.
    # 对图像应用验证集预处理流程。
    img = data_transforms['val'](img)

    # Add batch dimension: [C, H, W] -> [1, C, H, W].
    # 增加 batch 维度：从 [通道, 高, 宽] 变成 [1, 通道, 高, 宽]。
    img = img.unsqueeze(0)

    # Move image tensor to GPU or CPU.
    # 将图像 Tensor 移动到 GPU 或 CPU。
    img = img.to(device)

    # Disable gradient computation during inference.
    # 推理阶段不需要计算梯度。
    with torch.no_grad():

        # Forward pass.
        # 前向传播，得到模型输出。
        outputs = model(img)

        # Get predicted class index.
        # 获取预测类别编号。
        _, preds = torch.max(outputs, 1)

        # Create a subplot.
        # 创建一个子图。
        ax = plt.subplot(2, 2, 1)

        # Hide axis.
        # 隐藏坐标轴。
        ax.axis('off')

        # Set title as predicted class name.
        # 设置标题为预测类别名称。
        ax.set_title(f'Predicted: {class_names[preds[0]]}')

        # Show the input image after moving it back to CPU.
        # 将图像移回 CPU 后显示。
        imshow(img.cpu().data[0])

        # Restore original model state.
        # 恢复模型原来的训练/评估状态。
        model.train(mode=was_training)


# ============================================================
# 12. Run single image prediction
# 12. 执行单张图像预测
# ============================================================

# Define the test image path.
# 定义测试图像路径。
test_img_path = 'hymenoptera_data/val/bees/72100438_73de9f17af.jpg'

# Run prediction on the test image.
# 对测试图像进行预测。
visualize_model_predictions(model_conv, img_path=test_img_path)

# Turn off matplotlib interactive mode.
# 关闭 matplotlib 交互模式。
plt.ioff()

# Show all remaining plots.
# 显示所有剩余图像窗口。
plt.show()

print(f'训练结束')