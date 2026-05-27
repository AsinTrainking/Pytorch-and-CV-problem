# transfer_learning_template.py
# Standard PyTorch CV template for image classification / transfer learning

import os
import time
import copy
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler


# ============================================================
# 1. Basic configuration
# ============================================================

DATA_DIR = "hymenoptera_data"
OUTPUT_DIR = "outputs"

MODEL_NAME = "resnet18"
NUM_EPOCHS = 25
BATCH_SIZE = 8
NUM_WORKERS = 0          # Use 0 on Windows/Jupyter to avoid multiprocessing error
LEARNING_RATE = 0.001
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1

FEATURE_EXTRACT = False
# False: fine-tune the whole ResNet18
# True: freeze backbone and train only the final classifier

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_resnet18_checkpoint.pth")
HISTORY_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")


# ============================================================
# 2. Reproducibility and device
# ============================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True

    return device


# ============================================================
# 3. Data preparation
# ============================================================

def get_data_transforms():
    """
    Standard ImageNet-style transforms for transfer learning.

    Training:
        random crop + random horizontal flip + normalization

    Validation:
        deterministic resize + center crop + normalization
    """

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
    }

    return data_transforms


def build_dataloaders(data_dir, batch_size, num_workers):
    """
    Expected folder structure:

    hymenoptera_data/
        train/
            ants/
            bees/
        val/
            ants/
            bees/
    """

    data_transforms = get_data_transforms()

    image_datasets = {
        phase: datasets.ImageFolder(
            root=os.path.join(data_dir, phase),
            transform=data_transforms[phase]
        )
        for phase in ["train", "val"]
    }

    dataloaders = {
        phase: torch.utils.data.DataLoader(
            image_datasets[phase],
            batch_size=batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        for phase in ["train", "val"]
    }

    dataset_sizes = {
        phase: len(image_datasets[phase])
        for phase in ["train", "val"]
    }

    class_names = image_datasets["train"].classes

    print("Dataset loaded successfully.")
    print(f"Classes: {class_names}")
    print(f"Training images: {dataset_sizes['train']}")
    print(f"Validation images: {dataset_sizes['val']}")

    return dataloaders, dataset_sizes, class_names, data_transforms


# ============================================================
# 4. Visualization helper
# ============================================================

def imshow(inp, title=None):
    """
    Display a Tensor image after unnormalization.
    """

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)


def show_training_batch(dataloaders, class_names):
    """
    Show one batch of training images.
    """

    inputs, classes = next(iter(dataloaders["train"]))
    out = torchvision.utils.make_grid(inputs)

    plt.figure(figsize=(8, 6))
    imshow(out, title=[class_names[x] for x in classes])
    plt.show()


# ============================================================
# 5. Model preparation
# ============================================================

def set_parameter_requires_grad(model, feature_extract):
    """
    If feature_extract=True, freeze all existing parameters.
    Only the newly replaced classifier layer will be trainable.
    """

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def build_model(num_classes, feature_extract=False):
    """
    Build a pretrained ResNet18 model.

    If feature_extract=False:
        fine-tune the whole model.

    If feature_extract=True:
        freeze the backbone and train only the final fc layer.
    """

    model = models.resnet18(weights="IMAGENET1K_V1")

    set_parameter_requires_grad(model, feature_extract)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def get_trainable_parameters(model):
    """
    Return only parameters that require gradients.
    This works for both full fine-tuning and feature-extractor mode.
    """

    params_to_update = [param for param in model.parameters() if param.requires_grad]

    print("Trainable parameter groups:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    return params_to_update


# ============================================================
# 6. Training and validation
# ============================================================

def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    device,
    checkpoint_path,
    class_names,
    num_epochs=25
):
    """
    Standard PyTorch training loop.

    Key features:
        - train/eval mode switching
        - torch.set_grad_enabled()
        - validation after every epoch
        - best checkpoint saved by validation accuracy
        - training history returned
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase:5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save({
                    "epoch": epoch + 1,
                    "model_name": MODEL_NAME,
                    "model_state_dict": best_model_wts,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_acc,
                    "class_names": class_names,
                    "num_classes": len(class_names),
                    "feature_extract": FEATURE_EXTRACT,
                    "history": history
                }, checkpoint_path)

                print(f"Saved new best model to: {checkpoint_path}")

    time_elapsed = time.time() - since

    print("\nTraining complete.")
    print(f"Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    return model, history


# ============================================================
# 7. Plot training curves
# ============================================================

def plot_history(history, save_path=None):
    """
    Plot training/validation loss and accuracy curves.
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        loss_path = save_path.replace(".png", "_loss.png")
        plt.savefig(loss_path, dpi=300, bbox_inches="tight")
        print(f"Loss curve saved to: {loss_path}")

    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        acc_path = save_path.replace(".png", "_acc.png")
        plt.savefig(acc_path, dpi=300, bbox_inches="tight")
        print(f"Accuracy curve saved to: {acc_path}")

    plt.show()


# ============================================================
# 8. Visualize model predictions
# ============================================================

def visualize_model(model, dataloaders, class_names, device, num_images=6):
    """
    Show several validation images with predicted labels.
    """

    was_training = model.training
    model.eval()

    images_so_far = 0
    plt.figure(figsize=(8, 8))

    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            for j in range(inputs.size(0)):
                images_so_far += 1

                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")

                pred_idx = preds[j].item()
                true_idx = labels[j].item()

                ax.set_title(
                    f"Pred: {class_names[pred_idx]}\nTrue: {class_names[true_idx]}"
                )

                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    return

    model.train(mode=was_training)
    plt.show()


# ============================================================
# 9. Single image inference
# ============================================================

def predict_single_image(model, img_path, data_transforms, class_names, device):
    """
    Run inference on one custom image.
    """

    if not os.path.exists(img_path):
        print(f"Image does not exist: {img_path}")
        return

    was_training = model.training
    model.eval()

    img = Image.open(img_path).convert("RGB")
    img_tensor = data_transforms["val"](img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_class = class_names[pred.item()]
    confidence = conf.item()

    print(f"Image: {img_path}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {pred_class} ({confidence:.2%})")
    plt.show()

    model.train(mode=was_training)


# ============================================================
# 10. Load checkpoint for later inference
# ============================================================

def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load a saved checkpoint and rebuild the ResNet18 model.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    feature_extract = checkpoint.get("feature_extract", False)

    model = build_model(
        num_classes=num_classes,
        feature_extract=feature_extract
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    print(f"Classes: {class_names}")

    return model, class_names


# ============================================================
# 11. Main function
# ============================================================

def main():
    set_seed(42)

    device = get_device()

    dataloaders, dataset_sizes, class_names, data_transforms = build_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Optional: show one batch of training images
    show_training_batch(dataloaders, class_names)

    num_classes = len(class_names)

    model = build_model(
        num_classes=num_classes,
        feature_extract=FEATURE_EXTRACT
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # imbalnced datasets, using weighted loss can help
    '''def compute_class_weights(image_dataset, device):
        targets = image_dataset.targets
        class_counts = torch.bincount(torch.tensor(targets))
        
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_counts)
    
        return class_weights.to(device)'''


    params_to_update = get_trainable_parameters(model)

    optimizer = optim.SGD(
        params_to_update,
        lr=LEARNING_RATE,
        momentum=MOMENTUM
    )

    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_path=CHECKPOINT_PATH,
        class_names=class_names,
        num_epochs=NUM_EPOCHS
    )

    plot_history(history, save_path=HISTORY_PLOT_PATH)

    visualize_model(
        model=model,
        dataloaders=dataloaders,
        class_names=class_names,
        device=device,
        num_images=6
    )

    # Example single-image inference
    test_img_path = os.path.join(
        DATA_DIR,
        "val",
        "bees",
        "72100438_73de9f17af.jpg"
    )

    predict_single_image(
        model=model,
        img_path=test_img_path,
        data_transforms=data_transforms,
        class_names=class_names,
        device=device
    )


if __name__ == "__main__":
    main()