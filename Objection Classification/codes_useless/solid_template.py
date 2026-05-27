import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
# 引入工业界评估利器
from sklearn.metrics import classification_report, confusion_matrix

def get_data_loaders(data_dir, batch_size, num_workers=4):
    """标准的图像分类预处理管道 (ImageNet 标准)"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # 增加独立的盲测集管道
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 兼容处理可能没有 test 文件夹的情况
    phases = [d for d in ['train', 'val', 'test'] if os.path.exists(os.path.join(data_dir, d))]
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in phases}
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], 
            batch_size=batch_size, 
            shuffle=(x == 'train'), # 只有训练集需要 shuffle
            num_workers=num_workers,
            pin_memory=True # 开启锁页内存，加速 GPU 数据传输
        ) for x in phases
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    """带有完整日志记录和早停逻辑雏形的训练循环"""
    since = time.time()
    
    # 用于绘制曲线的日志字典
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    with TemporaryDirectory() as tempdir:
        best_model_param_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_param_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # 记录历史记录
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc)

                # 依据验证集表现保存最佳模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_param_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        # 加载最佳权重
        model.load_state_dict(torch.load(best_model_param_path, map_location=device))
        
    return model, history

def plot_training_history(history, save_path='training_curves.png'):
    """自动生成并保存 Loss/Acc 曲线，面试加分项"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")
    plt.close()

def evaluate_model(model, dataloader, class_names, device, phase='test'):
    """终极评估模块：输出混淆矩阵与精确率/召回率，展现专业度"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader[phase]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print(f"\n===== {phase.upper()} EVALUATION REPORT =====")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def main():
    # 1. 基础配置
    cudnn.benchmark = True
    data_dir = 'hymenoptera_data'
    batch_size = 8
    num_epochs = 25
    num_workers = 4 # 推荐非0，开启多线程提高读图效率
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 2. 获取数据管道
    dataloaders, dataset_sizes, class_names = get_data_loaders(data_dir, batch_size, num_workers)
    print("Class names:", class_names)
    print("Dataset sizes:", dataset_sizes)

    # 3. 构建模型 (这里演示最常用的迁移学习全微调或部分微调)
    model_conv = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 如果你想做完全微调(全网更新)，就不写 param.requires_grad = False
    # 如果你想做固定特征提取，就放开下面两行：
    # for param in model_conv.parameters():
    #     param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))
    model_conv = model_conv.to(device)

    # 4. 损失函数与优化器 (加入常见的 Momentum 动量)
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # 5. 启动训练
    model_conv, history = train_model(
        model_conv, criterion, optimizer_conv, exp_lr_scheduler, 
        dataloaders, dataset_sizes, device, num_epochs=num_epochs
    )

    # 6. 后处理与专业评估
    plot_training_history(history)
    
    # 优先使用盲测集评估，若没有则用验证集汇报最终指标
    eval_phase = 'test' if 'test' in dataloaders else 'val'
    evaluate_model(model_conv, dataloaders, class_names, device, phase=eval_phase)

    # 7. 模型保存
    torch.save(model_conv.state_dict(), 'resnet18_best_model.pth')
    print('Model weights saved successfully.')

if __name__ == '__main__':
    main()