import os
import time
import zipfile

import random
import urllib.request

import torch
import torchvision
import torch.utils.data

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.transforms import v2 as T

from torchvision.utils import draw_bounding_boxes


DATA_ROOT = 'data'

DATASET_NAME = 'PennFudanPed'

DATASET_ZIP_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
OUTPUT_DIR = "outputs_detection2"

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "fasterrcnn_resnet50_fpn_pennfudan.pth"
)

NUM_CLASSES = 2

BATCH_SIZE = 2

NUM_EPOCHS = 5

LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

LR_STEP_SIZE = 3
LR_GAMMA = 0.1

NUM_WORKERS = 0

TRAIN_RATIO = 0.8

SEED = 42

SCORE_THRESHOLD = 0.5

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if device.type =="cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return device

def download_and_extract_dataset():
    os.makedirs(DATA_ROOT,exist_ok=True)

    dataset_dir = os.path.join(DATA_ROOT,DATASET_NAME)

    zip_path = os.path.join(DATA_ROOT,"PennFudanPed.zip")

    if os.path.exists(dataset_dir):
        print(f"Dataset already exists: {dataset_dir}")
        return dataset_dir
    
    print ("Downloading dataset...")

    print(f"URL: {DATASET_ZIP_URL}")

    urllib.request.urlretrieve(DATASET_ZIP_URL,zip_path)

    print("Extracting dataset")
    with zipfile.ZipFile(zip_path,'r') as zip_ref:

        zip_ref.extractall(DATA_ROOT)

    print(f"Dataset extracted to : {dataset_dir}")

    return dataset_dir


def get_transform(train):
    transform = []

    if train:
        transform.append(T.RandomHorizontalFlip(p=0.5))

    transform.append(T.ToImage())
    transform.append(T.ToDtype(torch.float32,scale=True))
    return T.Compose(transform)


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self,root, transforms = None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root,"PNGImages")))

        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)

        mask = np.array(mask)

        obj_ids = np.unique(mask)

        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:,None,None]

        boxes = []

        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])

            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            boxes.append([xmin,ymin,xmax,ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((len(obj_ids),),dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        area = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])

        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)


        target = {
            "boxes": boxes,

            "labels": labels,

            "masks": masks,
            "image_id": image_id,
            "area": area,

            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)
        

    
def collate_fn(batch):
    return tuple(zip(*batch))




def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights = "DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


def train_one_epoch(model,optimizer, dataloader, device, epoch):
    model.train()

    total_loss = 0.0

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]

        targets = [
            {k: v.to(device) for k,v in t.items()}
            for t in targets
        ]

        loss_dict = model(images,targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()

        optimizer.step()

        total_loss +=losses.item()

        if batch_idx % 10 ==0:
            loss_items = {k: round(v.item(), 4) for k,v in loss_dict.items()}


            print (
                f"Epoch [{epoch}]"
                f"Batch [{batch_idx}/{len(dataloader)}]"
                f"Loss: {losses.item():.4f}"
                f"{loss_items}"
            )
    avg_loss = total_loss / len(dataloader)

    elapsed = time.time() - start_time

    print(f"Epoch [{epoch}] Avage Loss: {avg_loss:.4f}")

    print(f"Epoch [{epoch}] Time: {elapsed:.1f}s")

    return avg_loss


def run_inference(model, dataloader, device, score_threshold=0.5, max_images=3):

    model.eval()

    images_shown = 0

    for images, targets in dataloader:

        images_gpu = [img.to(device) for img in images]

        outputs = model(images_gpu)


        for img, output, target in zip(images, outputs, targets):

            img_uint8 = (img*255).to(torch.uint8)

            scores = output["scores"].cpu()

            keep = scores>= score_threshold

            boxes = output["boxes"].cpu()[keep]

            labels = output["labels"].cpu()[keep]

            scores = scores[keep]

            label_texts = [
                f"pedestrain {score:.2f}"
                for score in scores
            ]

            drawn = draw_bounding_boxes(
                image = img_uint8,
                boxes=boxes,
                labels=label_texts,
                width=3
            )

            plt.figure(figsize=(8,6))

            plt.imshow(drawn.permute(1,2,0))

            plt.axis("off")

            plt.title("Predicted Bounding boxes")

            plt.show()

            images_shown += 1

            if images_shown >= max_images:
                return
            


def save_checkpoint(model, optimizer, epoch, loss_history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "num_classes": NUM_CLASSES,
            "class_names": ["background", "Pedestrain"],
        },
        path
    )

    print(f"Chekpoint save to: {path}")

def load_chekpoint(path, device):

    checkpoint = torch.load(path, map_location=device)

    model = build_model(num_classes = checkpoint["num_classes"])

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    model.eval()
    print(f"Checkpoint loaded from: {path}")

    print(f"Class names: {checkpoint['class_names']}")

    return model


def plot_loss(loss_history):

    plt.figure(figsize=(7,5))

    plt.plot(
        range(1, len(loss_history)+1),
        loss_history,
        marker = "o"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Average Training Loss")
    plt.title("Object Detection Training Loss")

    plt.grid(True)

    plt.show()

# ============================================================
# 12. Main function
# 12. 主函数
# ============================================================
def main():

    set_seed(SEED)

    device = get_device()

    dataset_dir = download_and_extract_dataset()

    full_dataset = PennFudanDataset(
        root=dataset_dir,
        transforms=get_transform(train=True)
    )

    test_dataset = PennFudanDataset(
        root=dataset_dir,
        transforms=get_transform(train=False)
    )

    indices = torch.randperm(len(full_dataset)).tolist()

    train_size = int(TRAIN_RATIO * len(indices))

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)

    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle= True,
        num_workers=NUM_WORKERS,
        collate_fn= collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Testing images: {len(test_dataset)}")

    model = build_model(num_classes=NUM_CLASSES)

    model.to(device)

    param = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        param,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size= LR_STEP_SIZE,
        gamma= LR_GAMMA
    )

    loss_history = []

    for epoch in range(1, NUM_EPOCHS+1):

        avg_loss = train_one_epoch(
            model = model,
            optimizer = optimizer,
            dataloader = train_loader,
            device = device,
            epoch = epoch
        )

        loss_history.append(avg_loss)

        lr_scheduler.step()

        save_checkpoint(
            model = model,
            optimizer = optimizer,
            epoch = epoch,
            loss_history = loss_history,
            path=CHECKPOINT_PATH
        )

    plot_loss(loss_history)

    run_inference(
        model = model,
        dataloader = test_loader,
        device = device,
        score_threshold = SCORE_THRESHOLD,
        max_images = 3
    )


if __name__ == "__main__":
    # EN:
    #   This ensures main() only runs when this script is executed directly.
    #   It is especially important on Windows when using DataLoader.
    #
    # CN:
    #   这可以确保只有直接运行该脚本时才会执行 main()。
    #   在 Windows 使用 DataLoader 时尤其重要。
    main()








        











    
