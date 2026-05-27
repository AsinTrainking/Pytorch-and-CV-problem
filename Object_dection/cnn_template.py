import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
import torchvision.models as models

# =====================================================================
# 1. 数据集定义模块 (Mock Dataset for Target/Label Shape Demonstration)
# =====================================================================
class MockObjectDetectionDataset(Dataset):
    """
    经典数据集下载/参考地址 (Dataset Download Links for Reference):
    - PASCAL VOC (20 classes): http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    - MS COCO (80 classes): https://cocodataset.org/
    """
    def __init__(self, num_samples=16, num_classes=20, grid_size=7):
        # 初始化样本数量、类别数量和网格大小
        # Initialize the number of samples, number of classes, and grid size
        super(MockObjectDetectionDataset, self).__init__()
        self.num_samples = num_samples      # 样本总数 / Total number of samples
        self.num_classes = num_classes      # 类别总数 / Total number of classes
        self.grid_size = grid_size          # 网格划分大小 (例如 7x7) / Grid size (e.g., 7x7)
        
        # 定义图像预处理：转化为张量并缩放到 448x448
        # Define image transforms: Convert to Tensor and resize to 448x448
        self.transform = transforms.Compose([
            transforms.ToTensor(),          # 转化为 PyTorch 张量 / Convert to PyTorch Tensor
            transforms.Resize((448, 448))   # 缩放至标准目标检测输入尺寸 / Resize to standard object detection input size
        ])

    def __len__(self):
        # 返回数据集的样本总数
        # Return the total number of samples in the dataset
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟生成一张标准的随机 RGB 图像，尺寸为 [通道数, 高度, 宽度]
        # Simulate generating a standard random RGB image with shape [Channels, Height, Width]
        mock_image = torch.rand(3, 448, 448)  # 形状为 [3, 448, 448] / Shape is [3, 448, 448]
        
        # 标签空间大小：1个置信度(Confidence) + 4个坐标(x,y,w,h) + 类别数量(num_classes)
        # Label dimension size: 1 confidence + 4 coordinates (x,y,w,h) + number of classes
        label_dim = 1 + 4 + self.num_classes  # 计算单个网格的标签维度 / Calculate label dimension for a single grid
        
        # 初始化网格标签张量，全部填充为 0
        # Initialize the grid label tensor, filled entirely with zeros
        # 最终形状为 [网格高, 网格宽, 标签维度] / Final shape: [Grid_H, Grid_W, Label_Dim]
        mock_label = torch.zeros(self.grid_size, self.grid_size, label_dim)
        
        # 模拟在网格坐标 (3, 3) 的位置填入一个真实的目标物体
        # Simulate placing a ground-truth target object at grid coordinate (3, 3)
        mock_label[3, 3, 0] = 1.0  # 索引0代表置信度：1.0表示网格内存在物体 / Index 0 is confidence: 1.0 means object exists
        
        # 索引1到4代表边界框：[x_center, y_center, width, height] 的相对归一化比例
        # Indices 1 to 4 are bounding box: [x_center, y_center, width, height] relative normalized ratios
        mock_label[3, 3, 1:5] = torch.tensor([0.5, 0.5, 0.2, 0.3])
        
        # 索引5之后是类别的 One-Hot 编码：假设属于第0类
        # Index 5 onwards is One-Hot encoding for classes: Assume it belongs to class 0
        mock_label[3, 3, 5] = 1.0  # 激活第 0 类 / Activate class 0
        
        # 返回图像张量和对应的目标检测标签
        # Return the image tensor and the corresponding object detection label
        return mock_image, mock_label

# =====================================================================
# 2. 网络架构模块 (Backbone + Custom Object Detection Head)
# =====================================================================
class CNNObjectDetector(nn.Module):
    def __init__(self, num_classes=20, grid_size=7):
        # 构造函数：初始化网络层结构
        # Constructor: Initialize network layer architectures
        super(CNNObjectDetector, self).__init__()
        self.grid_size = grid_size          # 保存网格大小 / Save grid size
        self.num_classes = num_classes      # 保存类别数量 / Save number of classes
        
        # 载入预训练的 ResNet18 作为骨干网络提取图像特征 (面试加分项)
        # Load pre-trained ResNet18 as backbone to extract image features (Bonus for interview)
        # 2. 将原来的 resnet = models.resnet18(pretrained=True) 改为：
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 裁剪网络：通过 list() 提取 ResNet18 所有子层，并切片丢弃最后两层(平均池化层和全连接分类层)
        # Crop network: Extract all children layers of ResNet18 via list(), and slice to discard the last two layers (AvgPool and Linear classification)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 此时输出特征图大小为 [Batch, 512, 14, 14] / Output feature map shape is [Batch, 512, 14, 14]
        
        # 计算检测头全连接层所需的总输出特征数：网格数 * 网格数 * 单网格预测维度
        # Calculate the total number of output features needed for detection head: Grid * Grid * Prediction_Dim per grid
        output_dim = self.grid_size * self.grid_size * (1 + 4 + self.num_classes)
        
        # 构建检测头（Detection Head）：将骨干网络的高维特征映射为网格预测值
        # Construct the Detection Head: Map high-dimensional features from backbone to grid predictions
        self.detector_head = nn.Sequential(
            nn.Flatten(),                          # 将三维特征图展平成一维向量 / Flatten 3D feature maps into a 1D vector
            nn.Linear(512 * 14 * 14, 1024),       # 全连接层：从 512*14*14 降维至 1024 / Linear layer: Dimensionality reduction from 512*14*14 to 1024
            nn.LeakyReLU(0.1),                     # 激活函数：防止神经元坏死 / Activation function: Avoid dying neurons
            nn.Linear(1024, output_dim),           # 输出层：映射到最终的目标检测输出维度 / Output layer: Map to final object detection output dimension
            nn.Sigmoid()                           # 使用 Sigmoid 将所有预测值（坐标、置信度）压缩到 0~1 之间，便于稳定收敛 / Use Sigmoid to squash all predictions (coordinates, confidence) into 0~1 for stable convergence
        )

    def forward(self, x):
        # 前向传播逻辑：x 形状为 [Batch_Size, 3, 448, 448]
        # Forward propagation logic: x shape is [Batch_Size, 3, 448, 448]
        features = self.backbone(x)           # 输入骨干网络，提取特征 / Input into backbone, extract features -> Shape: [B, 512, 14, 14]
        out = self.detector_head(features)    # 输入检测头，得到一维长向量 / Input into detection head, get a 1D long vector -> Shape: [B, output_dim]
        
        # 关键操作：将一维向量重新编排(Reshape)为具有网格空间概念的四维张量
        # Crucial step: Reshape the 1D vector into a 4D tensor with grid spatial concepts
        # 变换后的形状为 [Batch_Size, Grid_H, Grid_W, 1+4+Classes]
        # Reshaped shape: [Batch_Size, Grid_H, Grid_W, 1+4+Classes]
        out = out.view(-1, self.grid_size, self.grid_size, 1 + 4 + self.num_classes)
        return out                            # 返回预测结果 / Return the prediction result

# =====================================================================
# 3. 多任务损失函数模块 (Multi-task Loss Function)
# =====================================================================
class ObjectDetectionLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        # 构造函数：初始化损失函数组件及权重系数
        # Constructor: Initialize loss function components and weight coefficients
        super(ObjectDetectionLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum") # 使用平方误差和(Sum of Squared Errors)作为基础损失 / Use Sum of Squared Errors (SSE) as the base loss
        self.lambda_coord = lambda_coord       # 坐标损失权重放大器（平衡定位能力）/ Coordinate loss weight amplifier (balances localization capability)
        self.lambda_noobj = lambda_noobj       # 无目标背景损失权重缩小器（防止负样本主导梯度）/ No-object background loss weight shrinker (prevents negative samples from dominating gradients)

    def forward(self, predictions, targets):
        # 前向损失计算：切片分离预测值和真实值 (通过最后维度的切片)
        # Forward loss calculation: Slice and separate predictions and ground truths (via the last dimension)
        
        pred_conf  = predictions[..., 0:1]     # 预测置信度 / Predicted Confidence -> [B, G, G, 1]
        pred_boxes = predictions[..., 1:5]     # 预测边界框坐标 / Predicted Bounding Box Coordinates -> [B, G, G, 4]
        pred_cls   = predictions[..., 5:]      # 预测类别概率 / Predicted Class Probabilities -> [B, G, G, Num_Classes]
        
        target_conf  = targets[..., 0:1]       # 真实置信度 / Ground-truth Confidence -> [B, G, G, 1]
        target_boxes = targets[..., 1:5]       # 真实边界框坐标 / Ground-truth Bounding Box Coordinates -> [B, G, G, 4]
        target_cls   = targets[..., 5:]        # 真实类别标签 / Ground-truth Class Labels -> [B, G, G, Num_Classes]
        
        # 掩码机制（Mask）：exists_box 作为权重乘数，只针对真实含有物体的网格（置信度为1）计算损失
        # Masking mechanism: 'exists_box' acts as a weight multiplier, ensuring losses are only calculated for grids that actually contain an object (confidence = 1)
        exists_box = target_conf               # 形状为 [B, G, G, 1]，有物体则为1，无物体则为0 / Shape [B, G, G, 1], 1 if object exists, 0 otherwise
        
        # (1) 坐标回归损失：通过与 exists_box 相乘，不含目标的网格直接被清零(不参与坐标梯度更新)
        # (1) Bounding Box Regression Loss: By multiplying with exists_box, grids without objects are zeroed out (not participating in coordinate gradient updates)
        loss_coord = self.mse(exists_box * pred_boxes, exists_box * target_boxes)
        
        # (2) 置信度损失：分为有物体网格的置信度误差，以及无物体背景网格的置信度误差（使用 1 - exists_box 过滤）
        # (2) Objectness Confidence Loss: Split into confidence error for object grids, and confidence error for background grids without objects (filtered using 1 - exists_box)
        loss_obj = self.mse(exists_box * pred_conf, exists_box * target_conf)                             # 有目标的置信度 / Object exists confidence
        loss_noobj = self.mse((1 - exists_box) * pred_conf, (1 - exists_box) * target_conf)               # 无目标的背景置信度 / Background without object confidence
        
        # (3) 类别分类损失：同样只针对包含物体的网格进行分类判定
        # (3) Classification Loss: Likewise, only classification of grids containing an object is evaluated
        loss_cls = self.mse(exists_box * pred_cls, exists_box * target_cls)
        
        # 多任务损失加权融合：根据 YOLO 经典论文公式，组装最终的总损失
        # Multi-task loss weighted fusion: Assemble the final total loss according to the classic YOLO paper formula
        total_loss = (
            self.lambda_coord * loss_coord     # 放大坐标定位损失 / Amplify coordinate localization loss
            + loss_obj                         # 包含物体的置信度损失 / Object confidence loss
            + self.lambda_noobj * loss_noobj   # 缩小背景无物体的置信度损失 / Shrink background no-object confidence loss
            + loss_cls                         # 分类损失 / Classification loss
        )
        return total_loss                      # 返回标量总损失 / Return the scalar total loss

# =====================================================================
# 4. 训练流程控制主入口 (Pipeline Execution)
# =====================================================================
def train_pipeline():
    # 声明网络超参数
    # Declare network hyperparameters
    NUM_CLASSES = 20                           # 类别数量 (例如 VOC 数据集为 20) / Number of classes (e.g., 20 for VOC dataset)
    GRID_SIZE = 7                              # 空间网格数 / Spatial grid size
    BATCH_SIZE = 4                             # 批处理大小 / Batch size
    EPOCHS = 3                                 # 训练总轮数 / Total training epochs
    
    # 动态检测当前硬件环境：优先使用 GPU CUDA，否则退回 CPU
    # Dynamically detect current hardware environment: Prefer GPU CUDA, otherwise fallback to CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current Execution Device: {DEVICE}") # 打印正在使用的计算设备 / Print the computing device being used

    # 1. 实例化自定义的目标检测数据集和 DataLoader 加载器
    # 1. Instantiate the custom object detection dataset and DataLoader
    dataset = MockObjectDetectionDataset(num_samples=16, num_classes=NUM_CLASSES, grid_size=GRID_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) # shuffle=True 开启数据洗牌 / shuffle=True enables data shuffling

    # 2. 实例化网络模型、多任务损失函数、以及 Adam 优化器
    # 2. Instantiate the network model, multi-task loss function, and Adam optimizer
    model = CNNObjectDetector(num_classes=NUM_CLASSES, grid_size=GRID_SIZE).to(DEVICE) # 将模型移至对应的显卡或CPU / Move model to corresponding GPU or CPU
    criterion = ObjectDetectionLoss()                                                 # 实例化损失函数 / Instantiate loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)                               # 优化器，学习率设为 0.0001 / Optimizer, learning rate set to 0.0001

    # 3. 开启标准的深度学习模型训练循环
    # 3. Start the standard deep learning model training loop
    model.train()                                                                     # 将模型设置为训练模式(激活 Dropout 和 BatchNorm) / Set model to training mode (activates Dropout & BatchNorm)
    for epoch in range(EPOCHS):                                                       # 循环每一轮 Epoch / Loop through each epoch
        epoch_loss = 0.0                                                              # 初始化当前 Epoch 的累加损失 / Initialize cumulative loss for current epoch
        for images, labels in dataloader:                                             # 遍历批次数据 / Iterate through batch data
            images = images.to(DEVICE)                                                # 将输入图片移至计算设备 / Move input images to computing device
            labels = labels.to(DEVICE)                                                # 将真实标签移至计算设备 / Move ground-truth labels to computing device
            
            # 前向传播：模型预测输出
            # Forward propagation: Model predicts output
            outputs = model(images)                                                   # 获取预测张量 / Get prediction tensor -> [B, G, G, 1+4+C]
            loss = criterion(outputs, labels)                                         # 计算当前批次的多任务损失 / Calculate multi-task loss for current batch
            
            # 反向传播与权重参数梯度更新
            # Backward propagation and weight parameter gradient updates
            optimizer.zero_grad()                                                     # 清空上一批次的残留梯度，防止梯度累加 / Clear residual gradients from previous batch to prevent gradient accumulation
            loss.backward()                                                           # 反向传播，计算当前梯度的偏导数 / Backward pass to compute partial derivatives of gradients
            optimizer.step()                                                          # 根据 Adam 算法规则更新网络权重参数 / Update network weight parameters based on Adam algorithm rules
            
            epoch_loss += loss.item()                                                 # 累加批次损失标量值 / Accumulate batch loss scalar value
            
        # 打印当前轮次的平均多任务组合损失
        # Print the average multi-task combined loss for the current epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}], Multi-task Combined Loss: {epoch_loss/len(dataloader):.4f}")
    
    print("Training process finished successfully!")                                  # 提示训练流程圆满完成 / Indicate training pipeline completed perfectly

if __name__ == "__main__":
    # 执行主训练流水线
    # Execute the main training pipeline
    train_pipeline()