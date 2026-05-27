## copy the smallest transformer architecture

import os

import time

import copy

import random

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from torchvision.utils import make_grid

from PIL import Image


# basic configuration

DATA_DIR = "hymenoptera_data"
OUTPUT_DIR = "OUPUT_MINI_VIT"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_mini_vit.pth")
HISTORY_PLOT_PATH = os.path.join(OUTPUT_DIR, "mini_vit_training_curves.png")

IMG_SIZE = 64
PATCH_SIZE = 8

IN_CHANNELS = 3
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 1
DIM_FEEDFORWARD = 128
DROPOUT = 0.1

BATCH_SIZE = 4
NUM_EPOCHS = 3

NUM_WORKERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
STEP_SIZE = 2
GAMMA = 0.5
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 
    if device.type =="cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    return device

def get_data_transforms():
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ]),
    }
    return data_transforms


def build_dataloader(data_dir, batch_size, num_workers):
    data_transforms = get_data_transforms()
    image_datasets = {
        phase: datasets.ImageFolder(
            root= os.path.join(data_dir,phase),
            transform=data_transforms[phase]
        )
        for phase in ["train","val"]
    }

    dataloaders = {
        phase: DataLoader(
            image_datasets[phase],
            batch_size=batch_size,
            shuffle = True if phase =="train" else False,
            num_workers = num_workers,
            pin_memory = True if torch.cuda.is_available() else False

        )
        for phase in ["train", "val"]        
    }
    dataset_sizes = {
        phase:len(image_datasets[phase])
        for phase in ["train","val"]
    }

    class_names = image_datasets["train"].classes
    print("Dataset loaded successfully.")  # EN: Confirm successful loading. CN: 打印数据集加载成功信息。
    print(f"Classes: {class_names}")  # EN: Print class names, e.g., ants and bees. CN: 打印类别名，如 ants 和 bees。
    print(f"Training images: {dataset_sizes['train']}")  # EN: Print number of training images. CN: 打印训练图像数量。
    print(f"Validation images: {dataset_sizes['val']}")  # EN: Print number of validation images. CN: 打印验证图像数量。


    return dataloaders, dataset_sizes, class_names,data_transforms

#visualization

def unnormalize_image(tensor_img):
    img = tensor_img.cpu().numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def show_training_batch(dataloaders, class_names):
    inputs, labels = next(iter(dataloaders["train"]))
    grid = make_grid(inputs[:min(8, len(inputs))])
    plt.figure(figsize=(8, 6))
    plt.imshow(unnormalize_image(grid))
    plt.title([class_names[i] for i in labels[:min(8, len(labels))]])
    plt.axis("off")
    plt.show()


# 5. Mini Vision Transformer model

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size) * (img_size//patch_size)

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x

class MiniViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1,d_model))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward= dim_feedforward,
            dropout=dropout,
            activation = "gelu",
            batch_first= True

        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std = 0.02)
        nn.init.trunc_normal_(self.cls_token, std = 0.02)
        nn.init.trunc_normal_(self.head.weight, std = 0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self,x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size,-1,-1)
        x = torch.cat((cls_tokens,x), dim=1)
        x = x+self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)
        return logits
    

def build_model(num_classes):
    model = MiniViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        num_classes=num_classes,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT

    )   
    return model

#training

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        running_corrects += torch.sum(preds==labels.data).item()
        total_samples += inputs.size(0)
    epoch_loss = running_loss/total_samples
    epoch_acc = running_corrects/total_samples
    return epoch_loss,epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects +=torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc
    

def train_model(model, dataloaders, dataset_sizes, class_names, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr= LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size = STEP_SIZE,
        gamma = GAMMA
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    since = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("_" * 30)
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device = device
        )

        val_loss, val_acc = evaluate(
            model = model,
            dataloader=dataloaders["val"],
            criterion = criterion,
            device=device
        )

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc>best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_model_wts,
                    "class_names": class_names,
                    "num_classes":len(class_names),
                    "img_size":IMG_SIZE,
                    "patch_size":PATCH_SIZE,
                    "d_model": D_MODEL,
                    "history":history,
                    "best_val_acc":best_acc
                },
                CHECKPOINT_PATH
            )
            print(f"Saved best checkpoint to: {CHECKPOINT_PATH}")

    elapsed = time.time()-since
    print("\nTraining complete.")  # EN: Print training complete. CN: 打印训练完成。
    print(f"Training time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s")  # EN: Print training time. CN: 打印训练时间。
    print(f"Best validation accuracy: {best_acc:.4f}")  # EN: Print best validation accuracy. CN: 打印最佳验证准确率。
    model.load_state_dict(best_model_wts)
    return model, history


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
    print(f"Saved plots to: {loss_path} and {acc_path}")  # EN: Print saved plot paths. CN: 打印曲线保存路径

def visualize_model(model, dataloaders, class_names, device, num_images = 6):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(8,8))
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


# 9. Single image inference 
def predict_single_image(model, img_path, data_transform, class_names, device):
    if not os.path.exists(img_path):
        print(f"Image does not exiist: {img_path}")
        return
    model.eval()
    img = Image.open(img_path).convert("RGB")
    img_tensor = data_transform["val"](img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits,dim=1)
        conf,pred = torch.max(probs,dim=1)
    
    pred_name = class_names[pred.item()]
    confidence = conf.item()
    # EN: Convert confidence to Python float. CN: 将置信度转换成 Python 浮点数。
    print(f"Image: {img_path}")  # EN: Print image path. CN: 打印图片路径。
    print(f"Predicted class: {pred_name}")  # EN: Print predicted class. CN: 打印预测类别。
    print(f"Confidence: {confidence:.4f}")  # EN: Print confidence. CN: 打印置信度。
    plt.figure(figsize=(5, 5))  # EN: Create figure. CN: 创建画布。
    plt.imshow(img)  # EN: Show original image. CN: 显示原始图像。
    plt.axis("off")  # EN: Hide axes. CN: 隐藏坐标轴。
    plt.title(f"Predicted: {pred_name} ({confidence:.2%})")  # EN: Show prediction title. CN: 显示预测结果标题。
    plt.show()  # EN: Display image. CN: 显示图像



# 10. Main function

def main():
    set_seed(SEED)
    device =get_device()
    dataloaders, dataset_sizes, class_names, data_transforms = build_dataloader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    show_training_batch(dataloaders, class_names)
    num_classes = len(class_names)
    model = build_model(num_classes=num_classes)

    model = model.to(device)
    print(model)
    model, history = train_model(
        model=model,
        dataloaders = dataloaders,
        dataset_sizes=dataset_sizes,
        class_names=class_names,
        device=device
    )
    plot_history(history)
    visualize_model(
        model=model,
        dataloaders=dataloaders,
        class_names=class_names,
        device=device,
        num_images=6
    )
    test_img_path = os.path.join(DATA_DIR,"val",class_names[0],os.listdir(os.path.join(DATA_DIR,"val",class_names[0]))[0])

    predict_single_image(
        model=model,
        img_path=test_img_path,
        data_transform=data_transforms,
        class_names=class_names,
        device=device
    )



if __name__ == "__main__":
    main()