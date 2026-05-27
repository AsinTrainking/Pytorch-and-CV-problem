# transfer_learning_template.py
# Standard PyTorch CV template for image classification / transfer learning
# 标准的 PyTorch 计算机视觉图像分类/迁移学习模板

import os  # Import OS module for path operations | 导入操作系统模块用于路径操作
import time  # Import time module to benchmark training duration | 导入时间模块用于计算训练耗时
import copy  # Import copy for deep copying model weights | 导入复制模块用于深拷贝模型权重
import torch  # Import core PyTorch library | 导入 PyTorch 核心库
import torchvision  # Import torchvision for CV datasets and utilities | 导入 torchvision 用于计算机视觉数据集与工具
import torch.nn as nn  # Import neural network modules as nn | 导入神经网络模块并简写为 nn
import torch.optim as optim  # Import optimization algorithms | 导入优化算法模块
import torch.backends.cudnn as cudnn  # Import cuDNN backends for GPU acceleration | 导入 cuDNN 后端以配置 GPU 加速
import numpy as np  # Import NumPy for numerical and matrix operations | 导入 NumPy 用于数值与矩阵计算
import matplotlib.pyplot as plt  # Import pyplot for plotting training curves | 导入 pyplot 用于绘制训练曲线

from PIL import Image  # Import Image for loading custom image files | 导入 PIL 的 Image 用于加载自定义图像文件
from torchvision import datasets, models, transforms  # Import specific CV components | 导入特定的视觉数据集、模型和图像变换模块
from torch.optim import lr_scheduler  # Import learning rate schedulers | 导入学习率调度器模块

# ============================================================
# 1. Basic configuration | 基础配置
# ============================================================

DATA_DIR = "hymenoptera_data"  # Path to the dataset directory | 数据集目录的路径
OUTPUT_DIR = "outputs"  # Directory to save checkpoints and plots | 存放模型权重和图表的输出目录

MODEL_NAME = "resnet18"  # Name of the backbone architecture | 主干网络架构的名称
NUM_EPOCHS = 25  # Total number of training epochs | 总训练轮数
BATCH_SIZE = 8  # Mini-batch size for training and validation | 训练和验证的批大小
NUM_WORKERS = 0  # 0 means single-process loading, avoids Windows errors | 0代表单进程加载，避免 Windows 环境多进程报错
LEARNING_RATE = 0.001  # Initial learning rate for the optimizer | 优化器的初始学习率
MOMENTUM = 0.9  # Momentum factor for SGD optimizer | SGD 优化器的动量因子
STEP_SIZE = 7  # Decay the learning rate every X epochs | 每隔 X 轮衰减一次学习率
GAMMA = 0.1  # Multiplicative factor of learning rate decay | 学习率衰减的乘法系数

FEATURE_EXTRACT = False  # False: fine-tune all layers; True: freeze backbone, train linear head | False:微调全网; True:冻结主干只训分类头

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_resnet18_checkpoint.pth")  # Best model save path | 最佳模型保存路径
HISTORY_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")  # Training curve save path | 训练曲线图保存路径

# ============================================================
# 2. Reproducibility and device | 可复现性与设备配置
# ============================================================

def set_seed(seed=42):  # Define function to fix random seeds | 定义固定随机种子的函数
    torch.manual_seed(seed)  # Set CPU seed for PyTorch random operations | 设置 PyTorch 在 CPU 上的随机种子
    torch.cuda.manual_seed_all(seed)  # Set GPU seeds for all graphics cards | 设置所有 GPU 的 PyTorch 随机种子
    np.random.seed(seed)  # Set random seed for NumPy operations | 设置 NumPy 的随机种子

def get_device():  # Define function to configure compute device | 定义获取计算设备的函数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Select CUDA if available, else CPU | 优先选择 GPU，否则使用 CPU
    print(f"Using {device} device")  # Print current active device | 打印当前启用的设备

    if device.type == "cuda":  # Check if the configured device is a GPU | 如果当前计算设备是 GPU
        print(f"GPU name: {torch.cuda.get_device_name(0)}")  # Print graphics card hardware model | 打印显卡的具体硬件型号
        cudnn.benchmark = True  # Enable cuDNN auto-tuner for speed boost | 开启 cuDNN 自动优化以加速固定尺寸输入的卷积

    return device  # Return the configured device object | 返回配置好的设备对象

# ============================================================
# 3. Data preparation | 数据准备
# ============================================================

def get_data_transforms():  # Define function for standard ImageNet transforms | 定义获取标准 ImageNet 图像变换的函数
    data_transforms = {  # Dictionary holding transforms for different phases | 包含不同阶段图像变换的字典
        "train": transforms.Compose([  # Compose multiple training transforms sequentially | 组合多个训练集图像变换操作
            transforms.RandomResizedCrop(224),  # Crop random area and resize to 224x224 | 随机裁剪区域并缩放到 224x224
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally | 随机对图像进行水平翻转
            transforms.ToTensor(),  # Convert PIL Image/ndarray to Tensor scaled to [0,1] | 将图像转换为 Tensor 并归一化到 [0,1]
            transforms.Normalize(  # Normalize tensor with ImageNet mean and std | 使用 ImageNet 的均值和标准差对张量进行标准化
                mean=[0.485, 0.456, 0.406],  # ImageNet channel-wise mean values | ImageNet 三通道的均值
                std=[0.229, 0.224, 0.225]  # ImageNet channel-wise standard deviations | ImageNet 三通道的标准差
            )  # End of Normalize | 标准化结束
        ]),  # End of train transform sequence | 训练集变换序列结束
        "val": transforms.Compose([  # Compose multiple validation transforms sequentially | 组合多个验证集图像变换操作
            transforms.Resize(256),  # Resize smaller edge to 256 while keeping aspect ratio | 保持宽高比将短边缩放到 256
            transforms.CenterCrop(224),  # Crop a deterministic 224x224 patch from the center | 从图像正中心裁剪固定的 224x224 区域
            transforms.ToTensor(),  # Convert validation image to PyTorch Tensor | 将验证集图像转换为 Tensor
            transforms.Normalize(  # Apply identical normalization as training | 应用与训练集完全相同的标准化参数
                mean=[0.485, 0.456, 0.406],  # Validation channel-wise mean | 验证集三通道均值
                std=[0.229, 0.224, 0.225]  # Validation channel-wise std | 验证集三通道标准差
            )  # End of Normalize | 标准化结束
        ]),  # End of val transform sequence | 验证集变换序列结束
    }  # End of dictionary | 字典结束

    return data_transforms  # Return the transformation pipelines | 返回构造好的数据变换字典

def build_dataloaders(data_dir, batch_size, num_workers):  # Function to construct PyTorch dataloaders | 建立 PyTorch 数据加载器的函数
    data_transforms = get_data_transforms()  # Fetch training and validation pipelines | 获取训练和验证的图像变换流程

    image_datasets = {  # Load datasets directly from arranged local folders | 直接从本地文件夹结构中加载数据集
        phase: datasets.ImageFolder(  # Map sub-folders into standard target classification classes | 将子文件夹自动映射为标准的分类类别
            root=os.path.join(data_dir, phase),  # Combine paths to reach train or val subfolders | 拼接路径指向 train 或 val 子文件夹
            transform=data_transforms[phase]  # Inject corresponding transform pipeline | 注入对应的图像变换流程
        )  # End of ImageFolder | ImageFolder 实例化结束
        for phase in ["train", "val"]  # Loop through training and validation phases | 循环遍历训练和验证两个阶段
    }  # End of dataset dict comprehension | 数据集字典推导式结束

    dataloaders = {  # Wrap datasets inside iterable multi-threaded loaders | 将数据集包装进可迭代的多线程加载器
        phase: torch.utils.data.DataLoader(  # Instantiating standard PyTorch DataLoader | 实例化标准的 PyTorch DataLoader
            image_datasets[phase],  # Pass the targeted dataset instance | 传入对应的目标数据集实例
            batch_size=batch_size,  # Set the number of samples processed per step | 设置每步处理的样本数量
            shuffle=True if phase == "train" else False,  # Shuffle train to randomize gradient descent, val stays ordered | 训练集打乱顺序以随机化梯度下降，验证集不打乱
            num_workers=num_workers,  # Define background threads for data loading | 设定数据加载的后台线程数
            pin_memory=True if torch.cuda.is_available() else False  # Page-lock memory for faster CPU-to-GPU tensor transfers | 锁页内存配置，若有 GPU 则加速 CPU 到 GPU 的数据搬运
        )  # End of DataLoader | DataLoader 实例化结束
        for phase in ["train", "val"]  # Loop through training and validation phases | 循环遍历训练和验证两个阶段
    }  # End of dataloader dict comprehension | 加载器字典推导式结束

    dataset_sizes = {  # Store the total number of images in each partition | 存储每个数据划分中的图像总总数
        phase: len(image_datasets[phase])  # Call len() to fetch sample count | 调用 len() 获取样本总数
        for phase in ["train", "val"]  # Loop through training and validation phases | 循环遍历训练和验证两个阶段
    }  # End of size dict comprehension | 数量字典推导式结束

    class_names = image_datasets["train"].classes  # Extract class strings from folder names | 从文件夹名称中提取出类别文本标签列表

    print("Dataset loaded successfully.")  # Print logs signaling success | 打印数据集加载成功日志
    print(f"Classes: {class_names}")  # Print all parsed category names | 打印解析出来的所有类别名称
    print(f"Training images: {dataset_sizes['train']}")  # Print count of training images | 打印训练集图片总数
    print(f"Validation images: {dataset_sizes['val']}")  # Print count of validation images | 打印验证集图片总数

    return dataloaders, dataset_sizes, class_names, data_transforms  # Return all structured entities | 返回所有构建好的数据实体

# ============================================================
# 4. Visualization helper | 可视化辅助工具
# ============================================================

def imshow(inp, title=None):  # Function to convert tensors to viewable image plots | 将张量转换为可视图像图表的函数
    inp = inp.numpy().transpose((1, 2, 0))  # Convert to numpy and reorder from [C, H, W] to [H, W, C] | 转换为 numpy 并由 [通道, 高, 宽] 调整为 [高, 宽, 通道]

    mean = np.array([0.485, 0.456, 0.406])  # Re-declare original ImageNet mean arrays | 重新声明原始 ImageNet 均值数组
    std = np.array([0.229, 0.224, 0.225])  # Re-declare original ImageNet std arrays | 重新声明原始 ImageNet 标准差数组

    inp = std * inp + mean  # Denormalize: multiply by std and add mean | 反标准化：乘以标准差并加上均值
    inp = np.clip(inp, 0, 1)  # Clip pixel bounds safely inside floating [0, 1] range | 将像素值安全地裁剪在浮点数 [0, 1] 边界内

    plt.imshow(inp)  # Pass the processed array to matplotlib renderer | 将处理好的数组传给 matplotlib 渲染器

    if title is not None:  # Check if titles/labels were supplied | 检查是否提供了标题/标签
        plt.title(title)  # Apply specified string onto plot canvas | 将指定的文本渲染在图表上方

    plt.pause(0.001)  # Pause briefly to ensure plot updates correctly | 短暂暂停以确保图表窗口正确刷新显示

def show_training_batch(dataloaders, class_names):  # Sample and plot a training batch | 采样并绘制一个训练批次的函数
    inputs, classes = next(iter(dataloaders["train"]))  # Pop one single mini-batch out from training loader | 从训练集加载器中弹出一个 mini-batch
    out = torchvision.utils.make_grid(inputs)  # Grid multiple batch images together into one single canvas | 将一个批次的多张图片网格化拼接成一张大图

    plt.figure(figsize=(8, 6))  # Set size dimensions for visualization window | 设置可视化窗口的尺寸大小
    imshow(out, title=[class_names[x] for x in classes])  # Plot grid tensor with string list of class labels | 绘制网格张量，并附上类别标签的文本列表
    plt.show()  # Display figure to screen | 将画布正式渲染到屏幕上

# ============================================================
# 5. Model preparation | 模型准备
# ============================================================

def set_parameter_requires_grad(model, feature_extract):  # Toggle gradient switches based on style | 根据模式切换梯度计算开关的函数
    if feature_extract:  # Trigger block only if frozen backbone mode is selected | 仅在选择了冻结主干网络模式时触发该块
        for param in model.parameters():  # Loop systematically through all internal layer weights | 系统地循环遍历模型的所有内部层权重
            param.requires_grad = False  # Shut down tracking to lock backbone weights | 关闭梯度追踪以锁定主干权重不参与训练

def build_model(num_classes, feature_extract=False):  # Instantiate backbone and adapt output dimension | 实例化主干网络并适配输出维度的函数
    model = models.resnet18(weights="IMAGENET1K_V1")  # Load official ResNet18 pre-trained on ImageNet1K | 加载官方在 ImageNet1K 上预训练好的 ResNet18 模型

    set_parameter_requires_grad(model, feature_extract)  # Route model through freezing logical gate | 将模型传入冻结逻辑网关进行权重锁定配置

    num_ftrs = model.fc.in_features  # Fetch incoming input dimension of old classifier head | 获取原有全连接分类头输入的特征维度
    model.fc = nn.Linear(num_ftrs, num_classes)  # Replace old fc layer with new random linear head matching num_classes | 用匹配当前目标类别数的新随机线性层替换旧的 fc 层

    return model  # Return mutated architecture | 返回修改好的模型架构

def get_trainable_parameters(model):  # Filter out un-updatable nodes | 过滤出不可更新的权重节点的函数
    params_to_update = [param for param in model.parameters() if param.requires_grad]  # Comprehend only active weights needing optimizations | 列表推导式筛选出所有需要计算梯度的活跃权重

    print("Trainable parameter groups:")  # Print log block header | 打印日志块头部
    for name, param in model.named_parameters():  # Traverse parameter names alongside references | 遍历网络所有权重的名称与其引用
        if param.requires_grad:  # Check if node survived filtering | 检查该权重节点是否在可训练白名单中
            print(f"  {name}")  # Output clean names of blocks slated for actual updates | 输出将被实际计算并更新的层名称

    return params_to_update  # Return updatable parameter list | 返回可训练的参数列表

# ============================================================
# 6. Training and validation | 训练与验证循环
# ============================================================

def train_model(  # Define complete training and checkpointing loop | 定义完整的模型训练与断点保存循环函数
    model,  # Model architecture reference | 模型架构引用
    dataloaders,  # Dataloaders dictionary | 数据加载器字典
    dataset_sizes,  # Dataset size dictionary | 数据集大小字典
    criterion,  # Loss metric objective | 损失函数准则
    optimizer,  # Optimization backend solver | 优化器后端求解器
    scheduler,  # Learning rate dynamic manager | 学习率动态管理器
    device,  # Hardware mapping target | 硬件映射目标设备
    checkpoint_path,  # Local saving coordinate | 本地保存路径
    class_names,  # Class text labels | 类别文本标签
    num_epochs=25  # Upper limit duration parameter | 训练的最大轮数参数
):  # End of function declaration | 函数声明结束
    since = time.time()  # Cache starting timestamp | 缓存训练开始的时间戳

    best_model_wts = copy.deepcopy(model.state_dict())  # Deepcopy initialized weight configurations | 深拷贝初始的模型权重字典作为存根
    best_acc = 0.0  # Setup placeholder tracker for validation accuracy top record | 为验证集准确率历史最高记录设置占位追踪器

    history = {  # Instantiate a dictionary to record metrics per epoch | 实例化一个字典用于记录每轮的性能指标
        "train_loss": [],  # Training loss list | 训练集损失记录列表
        "train_acc": [],  # Training accuracy list | 训练集准确率记录列表
        "val_loss": [],  # Validation loss list | 验证集损失记录列表
        "val_acc": []  # Validation accuracy list | 验证集准确率记录列表
    }  # End of history dict | 历史记录字典结束

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # Secure destination directory folder path safely | 安全地创建目标输出目录文件夹

    for epoch in range(num_epochs):  # Initiate standard outer loop traversing epochs | 开启遍历训练轮数的标准外层循环
        print(f"\nEpoch {epoch + 1}/{num_epochs}")  # Print current training round step log | 打印当前训练轮次的进度日志
        print("-" * 30)  # Print cosmetic divider line | 打印装饰用的分割线

        for phase in ["train", "val"]:  # Loop through training phase first then validation phase | 依次循环进入训练阶段与验证阶段
            if phase == "train":  # Check if phase state matches training | 检查当前阶段是否为训练
                model.train()  # Activate dropout and batch normalization updates | 将模型设为训练模式，激活 Dropout 和 BatchNorm 更新
            else:  # If phase matches validation state | 如果当前阶段是验证
                model.eval()  # Freeze statistics, lock behavior for inference | 将模型设为评估模式，冻结统计指标并固化行为

            running_loss = 0.0  # Reset interval loss accumulator back to zero | 将当前轮次的累计损失重置为零
            running_corrects = 0  # Reset interval correct hit counter back to zero | 将当前轮次的累计正确预测数重置为零

            for inputs, labels in dataloaders[phase]:  # Step iteratively through minibatches from current phase loader | 迭代遍历当前阶段加载器的每一个 mini-batch
                inputs = inputs.to(device)  # Migrating images tensor into targeted hardware space | 将图像张量搬运至目标硬件设备空间
                labels = labels.to(device)  # Migrating labels tensor into targeted hardware space | 将标签张量搬运至目标硬件设备空间

                optimizer.zero_grad()  # Erase previous gradients to avoid cumulative contamination | 擦除上一步的梯度以防累加污染

                with torch.set_grad_enabled(phase == "train"):  # Dynamically switch tracking engine on for train, off for val | 动态开关：训练时开启梯度追踪，验证时关闭
                    outputs = model(inputs)  # Pass tensors forward through active model to generate logits | 图像前向传播穿过网络模型生成类别原始得分 Logits
                    _, preds = torch.max(outputs, dim=1)  # Extract indices holding the highest scoring channels | 提取得分最高通道所对应的索引（即预测类别）
                    loss = criterion(outputs, labels)  # Compute cross-entropy divergence metric value | 计算交叉熵损失散度度量值

                    if phase == "train":  # Execute parameter adjustment updates only during training | 仅在训练阶段执行参数调整更新
                        loss.backward()  # Trigger backpropagation to compute layer derivatives | 触发反向传播计算各层参数的导数梯度
                        optimizer.step()  # Let optimizer step forward and shift weight floats | 让优化器根据梯度更新网络中的浮点数权重

                running_loss += loss.item() * inputs.size(0)  # Re-scale batch mean loss to absolute scale and accumulate | 将批次平均损失还原为绝对总损失并累加
                running_corrects += torch.sum(preds == labels.data)  # Track matching hits and increment cumulative counter | 统计预测正确的样本数并累加进计数器

            if phase == "train":  # Perform scheduling steps right after training phase loops close | 在训练阶段循环结束后立即调用调度器
                scheduler.step()  # Trigger scheduled adjustment on internal optimizer learning rates | 触发预设好的优化器学习率规律衰减调整

            epoch_loss = running_loss / dataset_sizes[phase]  # Divide absolute totals over split sample count | 用累加的绝对总损失除以当前阶段的样本总数
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]  # Compute ratio of successful hits as floating precision | 计算正确命中数占当前阶段总数的比例（浮点精度）

            history[f"{phase}_loss"].append(epoch_loss)  # Record current loss inside history structure | 将当前轮次损失记录进历史字典
            history[f"{phase}_acc"].append(epoch_acc)  # Record current accuracy inside history structure | 将当前轮次准确率记录进历史字典

            print(f"{phase:5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")  # Format status strings cleanly inside console logs | 将状态指标整齐地格式化输出到控制台日志中

            if phase == "val" and epoch_acc > best_acc:  # Evaluation evaluation score against historical summit records | 检查当前验证集准确率是否打破了历史最高纪录
                best_acc = epoch_acc  # Overwrite old summit record with new champion score | 用新的历史最高得分覆盖旧记录
                best_model_wts = copy.deepcopy(model.state_dict())  # Cache state snapshot weights safely into deepcopy storage | 将当前优秀的模型权重字典安全地拷贝快存起来

                torch.save({  # Construct structural dict to output rich diagnostic checkpoints | 构造结构化字典以输出信息丰富的全功能断点文件
                    "epoch": epoch + 1,  # Log current progress step count index | 记录当前训练步数索引
                    "model_name": MODEL_NAME,  # Attach explicit architecture name string | 附带明确的架构名称字符串
                    "model_state_dict": best_model_wts,  # Inject best weights binaries | 注入处于巅峰状态的模型权重二进制参数
                    "optimizer_state_dict": optimizer.state_dict(),  # Store optimizer learning state matrices | 存储优化器的学习状态矩阵
                    "scheduler_state_dict": scheduler.state_dict(),  # Keep decay phase metrics records intact | 保持学习率调度器的衰减阶段指标完好
                    "best_val_acc": best_acc,  # Save historical performance record float | 保存历史最高性能得分浮点数
                    "class_names": class_names,  # Embed label decoder keys lists inside files | 在文件中嵌入类别名称解码映射列表
                    "num_classes": len(class_names),  # Track dimension layouts definitions | 记录类别总数的维度定义
                    "feature_extract": FEATURE_EXTRACT,  # Keep architectural configuration settings tracking flags | 保留网络架构冻结策略的追踪配置标记
                    "history": history  # Enclose performance matrices track summaries inside package | 将当前的性能指标历史追踪汇总封装进包内
                }, checkpoint_path)  # Flush block to storage disk location | 将保存字典一次性持久化写入存储磁盘中

                print(f"Saved new best model to: {checkpoint_path}")  # Announce saving location inside pipeline output console | 在控制台中声明新最佳模型的存盘路径

    time_elapsed = time.time() - since  # Determine exact elapsed seconds duration span | 计算整个训练流程所耗费的精确秒数

    print("\nTraining complete.")  # Signal training conclusion inside outputs logs | 在输出日志中宣告训练结束
    print(f"Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")  # Print training duration time formatted | 格式化输出训练总共耗费的分钟与秒数
    print(f"Best validation accuracy: {best_acc:.4f}")  # Highlight maximum classification correctness metrics achieved | 高亮展示模型在验证集上达到的最高分类准确率

    model.load_state_dict(best_model_wts)  # Reload optimal snapshot weights back to functional model object | 将最完美的历史权重重新加载回可用的模型对象中

    return model, history  # Return trained entity coupled with history records | 返回训练完毕的模型实例以及完整的历史指标字典

# ============================================================
# 7. Plot training curves | 绘制训练曲线
# ============================================================

def plot_history(history, save_path=None):  # Function to render training convergence graphs | 渲染训练收敛图表的曲线展示函数
    epochs = range(1, len(history["train_loss"]) + 1)  # Form standard sequential steps range index vectors | 建立标准的连续轮数递增范围索引向量

    plt.figure(figsize=(8, 6))  # Initialize layout window dimensions for Loss graph | 初始化损失函数图表的画布尺寸
    plt.plot(epochs, history["train_loss"], label="Train Loss")  # Plot train loss values along epoch markers | 沿着轮次坐标绘制训练集损失数值
    plt.plot(epochs, history["val_loss"], label="Val Loss")  # Plot validation loss values along epoch markers | 沿着轮次坐标绘制验证集损失数值
    plt.xlabel("Epoch")  # Label x-axis coordinate as Epoch | 将横坐标轴标注为 Epoch（轮次）
    plt.ylabel("Loss")  # Label y-axis coordinate as Loss value | 将纵坐标轴标注为 Loss（损失）
    plt.title("Training and Validation Loss")  # Set plot description strings banner header | 设定图表上方的描述性标题
    plt.legend()  # Render descriptive string boxes to identify curves | 渲染标识不同曲线的图例说明框
    plt.grid(True)  # Draw grid grids lines underneath curve traces | 在曲线背后绘制网格辅助线

    if save_path is not None:  # Check if custom storage pathway location strings exist | 检查是否指定了合法的本地磁盘存盘路径字符串
        loss_path = save_path.replace(".png", "_loss.png")  # Append customized naming extensions indicators | 智能替换生成损失图表专属的命名后缀
        plt.savefig(loss_path, dpi=300, bbox_inches="tight")  # Save plot in high resolution with tight borders | 以300DPI高清分辨率紧凑保存损失图表至磁盘
        print(f"Loss curve saved to: {loss_path}")  # Output feedback confirmation string to logs terminal | 向日志终端输出损失曲线成功存盘的确认提示

    plt.show()  # Display Loss graphs onto popups visual displays windows | 在可视窗口中将损失函数图表展示出来

    plt.figure(figsize=(8, 6))  # Initialize layout window dimensions for Accuracy graph | 为准确率图表重新初始化画布尺寸
    plt.plot(epochs, history["train_acc"], label="Train Acc")  # Trace training classification hits trajectories across time | 绘制训练分类准确率随时间推移的走势迹线
    plt.plot(epochs, history["val_acc"], label="Val Acc")  # Trace validation classification hits trajectories across time | 绘制验证分类准确率随时间推移的走势迹线
    plt.xlabel("Epoch")  # Define horizontal coordinates definitions description labels | 定义横坐标意义描述标签
    plt.ylabel("Accuracy")  # Define vertical coordinates definitions description labels | 定义纵坐标意义描述标签
    plt.title("Training and Validation Accuracy")  # Assign title string banner headers safely | 安全地为准确率图表赋予标题文本
    plt.legend()  # Show descriptive legends boxes safely on charts layouts | 在图表布局中安全地显示说明性图例框
    plt.grid(True)  # Turn structural alignment guidelines mesh layer on | 开启网格对齐参考线图层

    if save_path is not None:  # Check if files save targets paths locations are available | 检查文件存储路径是否可用
        acc_path = save_path.replace(".png", "_acc.png")  # Build accuracy chart filenames strings parameters | 建立准确率图表的专属文件名参数字符串
        plt.savefig(acc_path, dpi=300, bbox_inches="tight")  # Export figures to local storage paths accurately | 将准确率图表高清导出到本地路径下
        print(f"Accuracy curve saved to: {acc_path}")  # Report save events updates to current workspace standard pipelines | 向当前工作区标准管道报告准确率图表保存事件

    plt.show()  # Flush visual buffers and display accuracy chart | 刷新显示缓存并展示准确率图表

# ============================================================
# 8. Visualize model predictions | 可视化模型预测结果
# ============================================================

def visualize_model(model, dataloaders, class_names, device, num_images=6):  # Sample and plot qualitative inference tests | 随机采样并绘制定性推理测试结果的函数
    was_training = model.training  # Fetch state properties flags to prevent runtime bugs | 获取模型当前所处的训练状态状态标记，防止运行时逻辑错误
    model.eval()  # Enforce evaluation constraints patterns onto weight nodes | 强制将评估模式约束施加到网络权重节点上

    images_so_far = 0  # Initialize layout cell positioning variables index counter | 初始化子图网格单元格定位变量的计数器
    plt.figure(figsize=(8, 8))  # Allocate layout box space for predictions display matrix | 为预测显示矩阵分配画布布局空间

    with torch.no_grad():  # Shut down tracking components down completely to preserve memory | 完全关闭梯度追踪组件以节省计算资源和显存
        for inputs, labels in dataloaders["val"]:  # Stream batch assets systematically via validation loaders | 顺次通过验证集加载器提取批次数据资产
            inputs = inputs.to(device)  # Route input tensors directly into active device buffers | 将输入张量分配至当前活跃的计算设备缓存中
            labels = labels.to(device)  # Route target labels directly into active device buffers | 将目标标签张量分配至当前活跃的计算设备缓存中

            outputs = model(inputs)  # Perform forwarding operations through layers blocks | 执行前向传播计算，输出分类 logits
            _, preds = torch.max(outputs, dim=1)  # Classify labels by choosing maximal logit positions indicators | 通过选择最大 logit 的位置索引来归纳预测类别

            for j in range(inputs.size(0)):  # Iterate across samples contained inside current mini-batch array slice | 循环遍历当前 mini-batch 数组切片中的每一个样本
                images_so_far += 1  # Increment internal subplot cells target location index trackers | 递增内部子图单元格的目标位置索引追踪器

                ax = plt.subplot(num_images // 2, 2, images_so_far)  # Segment matrix rows layouts dynamic configuration targets | 动态切分并定位子图矩阵的目标单元格布局
                ax.axis("off")  # Disable boundaries borders axes lines from final output displays | 在最终输出中隐去坐标轴边缘线

                pred_idx = preds[j].item()  # Extract numerical type scalars value from prediction tensors array | 从预测张量中提取出整型的标量数值索引
                true_idx = labels[j].item()  # Extract numerical type scalars value from target groundtruth arrays | 从真实标签张量中提取出整型的标量数值索引

                ax.set_title(  # Generate dynamic multi-line visual descriptions overhead text strings | 在当前子图上方生成动态的双行视觉描述文本
                    f"Pred: {class_names[pred_idx]}\nTrue: {class_names[true_idx]}"  # Format prediction vs ground truth mapping labels | 格式化展示预测类别与真实类别的映射标签
                )  # End of text assignment | 文本赋值结束

                imshow(inputs.cpu().data[j])  # Pull sample data back into CPU space and unpack configurations arrays | 将样本数据拉回 CPU 空间并调用 imshow 函数还原反标准化图像

                if images_so_far == num_images:  # Break out if visualization requirements threshold parameters are met | 如果展示的图片数量达到了设定的阈值上限，则准备退出
                    model.train(mode=was_training)  # Revert behavioral characteristics parameters back to preserved configurations | 将模型的行为特征模式恢复至原本保存的状态
                    plt.show()  # Display matrix layout compilation windows onto standard view | 将组合好的预测矩阵图表画布展示在标准视图中
                    return  # Terminate visualization processes pipelines explicitly | 显式终止可视化处理流程

    model.train(mode=was_training)  # Fallback safety reset configuration assignments | 兜底安全复原：将模型恢复至初始状态
    plt.show()  # Display canvas graphs outputs cleanly inside workspaces screens | 将画布结果清晰地展示在工作区屏幕中

# ============================================================
# 9. Single image inference | 单张图像推理
# ============================================================

def predict_single_image(model, img_path, data_transforms, class_names, device):  # Handle standard customized individual inference jobs | 处理标准自定义单张图片推理任务的函数
    if not os.path.exists(img_path):  # Run safety validations tests check to avoid crash bugs | 执行安全验证检查，防范路径不存在导致的崩溃错误
        print(f"Image does not exist: {img_path}")  # Log error diagnostics indicators clearly to alert operations | 清晰地打印错误诊断提示，警示用户检查路径
        return  # Terminate operation sequence prematurely | 提前终止操作序列

    was_training = model.training  # Cache current structural flags state conditions accurately | 精确缓存模型当前的训练状态标记
    model.eval()  # Switch network behavior variables directly into standard evaluation frameworks | 将网络行为变量切换至标准的评估模式下

    img = Image.open(img_path).convert("RGB")  # Read file directly and enforce clean 3-channel RGB pixel mapping configurations | 读取文件并强制转换为干净的三通道 RGB 像素色彩模式
    img_tensor = data_transforms["val"](img)  # Route files through validation pipeline directly | 让图像直接通过验证集变换流程进行缩放裁剪和标准化
    img_tensor = img_tensor.unsqueeze(0)  # Insert mock placeholder dimension at index 0 to simulate batch structures | 在索引 0 处插入伪批次维度以模拟批次结构 `[1, C, H, W]`
    img_tensor = img_tensor.to(device)  # Allocate matrix records on current hardware execution boards targets | 将构建好的单张图像张量分配给当前的硬件设备

    with torch.no_grad():  # Shut calculation tracking features down safely | 安全关闭梯度计算追踪功能
        outputs = model(img_tensor)  # Feedforward matrix through linear blocks arrays elements | 前向传播穿过网络，输出分类 logits
        probs = torch.softmax(outputs, dim=1)  # Softmax transformation convert scores to probabilities floats distributions | 经过 Softmax 变换将原始得分转换为合规的概率分布浮点数
        conf, pred = torch.max(probs, dim=1)  # Extract confidence floats coupled with maximum indices items outputs | 提取出最大概率的置信度浮点数与其对应的类别索引输出

    pred_class = class_names[pred.item()]  # Decode label string matching calculated results index mapping coordinates | 通过索引解码出对应的类别字符串标签文本
    confidence = conf.item()  # Extract python standard numeric floats items from tensor metrics objects | 从张量度量对象中提取出 Python 标准的数值浮点数项

    print(f"Image: {img_path}")  # Print files location traces log reports accurately | 精确打印当前推断的文件位置日志报告
    print(f"Predicted class: {pred_class}")  # Output evaluation results labels safely to tracking terminal | 安全地向终端输出模型预测出的类别标签结果
    print(f"Confidence: {confidence:.4f}")  # Display calculated accuracy score probabilities boundaries explicitly | 显式展示模型输出的该预测结果的置信度概率

    plt.figure(figsize=(5, 5))  # Create squarish canvas displays layout box spaces | 创建方形的画布展示空间
    plt.imshow(img)  # Put original un-manipulated image binaries on rendering buffers | 将最原始的未经变换的图片二进制文件放入渲染缓冲区
    plt.axis("off")  # Eliminate borders indicators strings and numerical values labels | 隐去边缘指示线以及刻度数值标签
    plt.title(f"Predicted: {pred_class} ({confidence:.2%})")  # Set display metrics text on plots structures overhead titles | 在图表上方设置带有置信度百分比的预测结果标题
    plt.show()  # Display windows cleanly onto display monitor interfaces screens | 将图像窗口干净地渲染在显示器屏幕界面中

    model.train(mode=was_training)  # Restore initial structural configurations assignments parameters back gracefully | 优雅地将模型的模式参数恢复至执行推理前的初始配置状态

# ============================================================
# 10. Load checkpoint for later inference | 载入断点用于后续推理
# ============================================================

def load_model_from_checkpoint(checkpoint_path, device):  # Reconstruct model from saved pth files binaries assets | 从保存的 pth 二进制资产文件中重构模型的函数
    checkpoint = torch.load(checkpoint_path, map_location=device)  # Read binary configurations blocks safely using target mapping directives | 使用目标映射指令安全读取断点字典

    class_names = checkpoint["class_names"]  # Re-extract serialized label arrays schemas tracking structures keys | 重新提取出序列化的类别标签映射数组结构
    num_classes = checkpoint["num_classes"]  # Re-extract dimensions settings specifications sizes boundaries | 重新提取出类别总数的维度尺寸定义
    feature_extract = checkpoint.get("feature_extract", False)  # Retrieve freezing architecture execution state flags configuration options | 检索架构冻结策略的执行状态标记配置选项

    model = build_model(  # Trigger rebuild sequence pipeline blocks structures definitions | 触发网络模型重构流程
        num_classes=num_classes,  # Inject dynamic target dimensions layouts setups metrics | 注入动态的目标类别维度参数
        feature_extract=feature_extract  # Pass architectural strategies parameter settings keys flags | 传入网络架构冻结策略参数标记
    )  # End of build_model call | build_model 调用结束

    model.load_state_dict(checkpoint["model_state_dict"])  # Overwrite random initialization arrays with saved pre-trained weights parameters | 用保存的训练好的权重参数覆盖掉网络随机初始化的数组
    model = model.to(device)  # Distribute weight matrices tensors arrays allocations onto computing unit boards grids | 将权重矩阵张量重新分配到指定的计算硬件设备上
    model.eval()  # Freeze graph logic parameters settings to accept deployment execution paths | 固化模型参数状态，准备进入部署执行路径下

    print(f"Loaded checkpoint from: {checkpoint_path}")  # Confirm files validation events tracing status logs reports | 确认并打印成功载入本地断点文件的状态日志报告
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")  # Report historical maximum classification hits benchmarks parameters | 报告断点文件中记录的历史最高验证集准确率指标
    print(f"Classes: {class_names}")  # List decoded string maps names categories parameters schemas arrays | 列出解码出的类别标签数组模式内容

    return model, class_names  # Return active computational structures references paths blocks objects | 返回激活的模型结构引用对象与对应的类别名称列表

# Imbalanced datasets helper section | 不平衡数据集处理辅助函数小结块（被注释部分）
'''def compute_class_weights(image_dataset, device):  # Define placeholder function block tracking class metrics calculations | 预留计算不平衡类别权重分数的函数块
    targets = image_dataset.targets  # Slice groundtruth label arrays rows parameters records listings out | 提取出整个数据集的真实标签列表记录
    class_counts = torch.bincount(torch.tensor(targets))  # Count occurrences frequencies across targets coordinates elements arrays | 统计各个类别索引标签在数组中出现的频次
    
    class_weights = 1.0 / class_counts.float()  # Compute reciprocal ratios fractions scores parameters elements values | 计算频次倒数，频次越低的类别将获得越高的权重分数
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # Standardize weight vectors elements boundaries spaces ratios | 归一化权重向量并使其均值为1，保持数值分布合理

    return class_weights.to(device)'''  # Dispatch calculation outputs tensors schemas elements back safely onto hardware | 将计算完毕的类别权重张量安全分发回目标设备并返回

# ============================================================
# 11. Main function | 主程序入口函数
# ============================================================

def main():  # Define system orchestrator main function block sequence pipelines structures | 定义系统总调度主函数流程
    set_seed(42)  # Lock global seed configurations variables across workspaces environments securely | 在整个工作区环境中安全锁定全局随机种子变量为 42

    device = get_device()  # Analyze system backends properties and fetch active target computational nodes | 分析系统硬件环境并获取当前的活跃目标计算设备

    dataloaders, dataset_sizes, class_names, data_transforms = build_dataloaders(  # Construct structural processing frameworks streams datasets units | 触发构建数据加载网络的基础处理框架组件
        data_dir=DATA_DIR,  # Pass raw assets file path directories roots parameters matrices strings | 传入数据集本地文件路径目录字符串参数
        batch_size=BATCH_SIZE,  # Configure processing volumes capacities indicators benchmarks constants | 配置批大小处理容量常数指标
        num_workers=NUM_WORKERS  # Pass parallel pipelines workers allocations quotas threads indexes | 传入多线程并行的工作线程数配额
    )  # End of parameters configuration bindings | 参数配置绑定结束

    # Optional: show one batch of training images | 可选步骤：采样并展示一个训练批次的图像可视化结果
    show_training_batch(dataloaders, class_names)  # Sample batch blocks arrays pipelines visually and display to display monitors screens | 采样一个批次并在显示器屏幕上执行可视化渲染

    num_classes = len(class_names)  # Count absolute total target classifications categories items configurations arrays boundaries | 统计目标分类任务的绝对总类别项数量界限

    model = build_model(  # Trigger model architecture components allocation routines frameworks structures mappings | 触发模型架构组件构建流程
        num_classes=num_classes,  # Map terminal classifications layer dimensions boundaries to target configurations | 将终点分类层的输出维度边界映射为目标分类数
        feature_extract=FEATURE_EXTRACT  # Route initialization pathways according to parameters decisions settings targets | 根据设定的参数决定是否执行权重冻结
    )  # End of architectural initialization mapping | 架构初始化映射结束

    model = model.to(device)  # Push initialized layers matrices blocks nodes over into active hardware device arrays | 将初始化的层矩阵节点参数全部推入到活跃的硬件设备中

    criterion = nn.CrossEntropyLoss()  # Construct standard Multi-Class CrossEntropy loss calculator objectives modules | 实例化经典的多分类交叉熵损失函数计算器模块

    '''class_weights = compute_class_weights(  # Optional section: trigger imbalance weights solver modules arrays formulas | 可选段落：激活不平衡权重求解器模块
        image_dataset=dataloaders["train"].dataset,  # Point solver directly to active underlying databases lists | 将求解器直接指向当前底层的活动数据集列表
        device=device  # Attach calculation matrix targets directly onto hardware space execution areas | 将计算矩阵直接分配给硬件设备执行空间
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights) '''  # Inject calculated balanced class weights tensors arrays inside calculation criteria modules | 将计算出的平衡类别权重张量注入损失函数计算器中

    params_to_update = get_trainable_parameters(model)  # Inspect structural network layers graph nodes and extract active fields entries list | 检查网络层结构图节点并过滤提取出活跃的可训练参数列表

    optimizer = optim.SGD(  # Instantiate Stochastic Gradient Descent optimization backend engine solvers algorithms models | 实例化经典的随机梯度下降（SGD）优化器后端求解引擎
        params_to_update,  # Bind filtered active editable node layers weight parameters sequences listings | 绑定经过过滤的活跃可训练权重参数序列列表
        lr=LEARNING_RATE,  # Pass initial learning rates floats tuning scalars increments options | 传入初始学习率步长浮点调整标量
        momentum=MOMENTUM  # Inject inertia speed acceleration tracking constant adjustments markers scalars | 注入动量惯性追踪常数标量参数
    )  # End of optimizer backend engines configuration assignments bindings | 优化器引擎配置参数绑定结束

    scheduler = lr_scheduler.StepLR(  # Instantiate Step decay learning rate scheduling controller managers components architectures | 实例化阶梯式衰减学习率调度控制器组件
        optimizer,  # Link scheduling controllers directly onto active optimization backend solvers engines targets | 将调度控制器绑定到当前的优化器求解器对象上
        step_size=STEP_SIZE,  # Bind durations intervals decay benchmarks step counts targets elements integers | 绑定执行衰减的周期轮数间隔整型参数
        gamma=GAMMA  # Pass multiplicative multipliers modifiers fraction decay coefficients floats variables | 传入乘法衰减系数浮点数变量参数
    )  # End of scheduling engine modules installations allocations assignments | 调度引擎模块配置分配结束

    model, history = train_model(  # Fire execution sequence pipelines loops to perform models training and evaluation tracks | 启动执行序列管道循环以执行核心的模型训练与评估任务
        model=model,  # Pass model structures graph variables arrays configurations references mappings | 传入模型结构图变量数组配置引用
        dataloaders=dataloaders,  # Pass iterable data asset stream systems networks interfaces structures loaders | 传入可迭代的数据资产流加载器系统
        dataset_sizes=dataset_sizes,  # Pass databases sizing tracking matrices records schemas indices parameters | 传入数据集大小追踪记录字典参数
        criterion=criterion,  # Pass mathematical optimization objective calculator solvers function modules blocks | 传入损失函数目标计算器函数模块
        optimizer=optimizer,  # Pass gradient optimization solver backend solvers engine routines systems matrices | 传入梯度优化求解器后端引擎系统
        scheduler=scheduler,  # Pass stepping step adjustment decay rules controlling managers objects frameworks | 传入步进动态衰减规则控制管理器对象
        device=device,  # Pass compute runtime execution targeted board nodes platforms grids targets | 传入计算运行时执行的目标硬件平台设备
        checkpoint_path=CHECKPOINT_PATH,  # Pass files save destination paths directories strings records entries coordinates | 传入断点文件本地存盘路径字符串参数
        class_names=class_names,  # Pass lists collections schemas definitions strings category arrays labels entries | 传入类别名称定义的字符串标签数组列表
        num_epochs=NUM_EPOCHS  # Pass numerical iteration duration upper boundaries limit integer variables constants | 传入数值迭代轮数上限整型变量常数
    )  # End of model training pipelines workflows orchestration loops | 模型训练工作流编排循环结束

    plot_history(history, save_path=HISTORY_PLOT_PATH)  # Render diagnostics charts layouts curves profiles graphs and export output assets safely | 渲染模型收敛诊断图表曲线并安全导出本地图像资产

    visualize_model(  # Run qualitative visualization inspection pipelines to print sample output instances results matrix | 运行定性可视化检查流程以输出样例推理矩阵图
        model=model,  # Pass fine-tuned model entities models graphs structures arrays parameters mappings | 传入微调完毕的模型实体架构参数
        dataloaders=dataloaders,  # Pass dataloaders data streams structures packages channels providers devices | 传入数据加载器流通道组件包
        class_names=class_names,  # Pass string mapping categories collections layouts schemas profiles elements listings | 传入类别名称字符串映射集列表
        device=device,  # Pass current computational execution runtime target hardware boards setups platforms | 传入当前计算执行运行时的目标硬件平台设备
        num_images=6  # Limit total maximum qualitative samples size configurations allocations items indicators | 限制进行定性检查渲染的最大样本图像总数
    )  # End of verification plotting visualizations matrices layouts routines configurations | 验证可视化矩阵布局流程结束

    # Example single-image inference | 示例单张自定义图片推理路径组合
    test_img_path = os.path.join(  # Synthesize target custom testing asset file path directory location strings entries | 拼接合成目标的自定义测试单张图片资产路径字符串
        DATA_DIR,  # Dataset path root location component directories | 数据集路径根位置目录组件
        "val",  # Targeted validation subdirectories section | 目标的验证集子目录选段
        "bees",  # Category name folder destination | 蜜蜂类别名称文件夹目的地
        "72100438_73de9f17af.jpg"  # Target files baseline filename identifiers names variables strings | 目标文件的具体文件名标识符字符串
    )  # End of path strings concatenation arrays pipelines pipelines mappings | 路径字符串拼接绑定结束

    predict_single_image(  # Trigger single image testing inferences pipeline sequences routines tasks calculations | 触发单张图像推断预测计算处理流程
        model=model,  # Pass fine-tuned active intelligence network engine graphs blocks components nodes | 传入微调好的活跃智网模型引擎组件节点
        img_path=test_img_path,  # Pass verified custom targets assets paths strings directories locations coordinates | 传入验证过的自定义目标单图资产路径字符串
        data_transforms=data_transforms,  # Pass standard scaling operations pipelines variables structures models layouts | 传入标准图像缩放标准化变换流程组件
        class_names=class_names,  # Pass decode keys records dictionaries lists collections mappings profiles | 传入类别解码文本字典映射映射列表
        device=device  # Pass computational execution target hardware platforms boards environments settings architectures | 传入计算执行目标硬件平台架构设备
    )  # End of custom independent inference process block workflows orchestration jobs | 自定义独立推理流程工作流编排任务结束

if __name__ == "__main__":  # Standard entry checking block condition statements to identify root triggers executions | 标准的脚本程序主入口执行检查条件语句
    main()  # Run the primary main functions sequence loop directly | 直接启动运行核心的主函数程序序列循环