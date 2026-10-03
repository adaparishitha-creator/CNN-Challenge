import argparse
import copy
import os
import random
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SPLIT_SEED = 0

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONVNEXT_REPO = "https://github.com/facebookresearch/ConvNeXt.git"

PRETRAINED_URL = (
    "https://dl.fbaipublicfiles.com/convnext/"
    "convnext_small_22k_224.pth"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================

def create_dataloaders(train_dir):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3)
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    # Dataset used only to determine classes and indices.
    base_dataset = datasets.ImageFolder(train_dir)

    if len(base_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(base_dataset.classes)}."
        )

    total_size = len(base_dataset)

    if total_size != 2400:
        print(
            f"Warning: notebook used 2400 training images, "
            f"but this directory contains {total_size}."
        )

    val_size = int(round(total_size * 0.20))
    train_size = total_size - val_size

    # Reproduce the original notebook's fixed train/validation split.
    generator = torch.Generator().manual_seed(SPLIT_SEED)

    permutation = torch.randperm(
        total_size,
        generator=generator
    ).tolist()

    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_full = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_full = datasets.ImageFolder(
        train_dir,
        transform=val_transform
    )

    train_dataset = Subset(
        train_full,
        train_indices
    )

    val_dataset = Subset(
        val_full,
        val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_loader,
        val_loader,
        base_dataset.classes
    )


# ============================================================
# CONVNEXT-SMALL MODEL
# ============================================================

def build_model(device):
    repo_dir = "ConvNeXt"

    if not os.path.exists(repo_dir):
        print("Downloading official ConvNeXt repository...")

        result = os.system(
            f'git clone --depth 1 "{CONVNEXT_REPO}" "{repo_dir}"'
        )

        if result != 0:
            raise RuntimeError(
                "Failed to clone the official ConvNeXt repository."
            )

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from models.convnext import convnext_small

    # The official ImageNet-22K checkpoint has a 21,841-class head.
    model = convnext_small(
        pretrained=False,
        in_22k=True,
        num_classes=21841
    )

    checkpoint_file = "convnext_small_22k_224.pth"

    if not os.path.exists(checkpoint_file):
        print(
            "Downloading ImageNet-22K pretrained "
            "ConvNeXt-Small checkpoint..."
        )

        urllib.request.urlretrieve(
            PRETRAINED_URL,
            checkpoint_file
        )

    print("Loading ImageNet-22K pretrained weights...")

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    # Replace ImageNet-22K classification head with 16-class head.
    in_features = model.head.in_features

    model.head = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    model = model.to(device)

    return model


# ============================================================
# VALIDATION
# ============================================================

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        running_loss / total,
        correct / total
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device
):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    mixup = v2.MixUp(
        alpha=0.2,
        num_classes=NUM_CLASSES
    )

    cutmix = v2.CutMix(
        alpha=1.0,
        num_classes=NUM_CLASSES
    )

    mix_transform = v2.RandomChoice([
        mixup,
        cutmix
    ])

    best_val_accuracy = 0.0
    best_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_seen = 0

        for images, labels in train_loader:
            images, mixed_labels = mix_transform(
                images,
                labels
            )

            images = images.to(
                device,
                non_blocking=True
            )

            mixed_labels = mixed_labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                mixed_labels
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * images.size(0)
            )

            total_seen += images.size(0)

        train_loss = (
            total_loss / total_seen
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            best_state = copy.deepcopy(
                model.state_dict()
            )

    # Restore best validation checkpoint.
    model.load_state_dict(best_state)

    return model, best_val_accuracy


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the selected ConvNeXt-Small "
            "CNN Challenge model."
        )
    )

    parser.add_argument(
        "--train-dir",
        type=str,
        default="train",
        help="Path to the training dataset."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="experiment6_best_model.pth",
        help="Output checkpoint path."
    )

    args = parser.parse_args()

    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {args.train_dir}"
        )

    set_random_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("======================================")
    print("CNN CHALLENGE - FINAL TRAINING RECIPE")
    print("======================================")
    print("Device:", device)
    print("Training directory:", args.train_dir)
    print("Architecture: ConvNeXt-Small")
    print("Pretraining: ImageNet-22K")
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Training seed:", SEED)
    print("Split seed:", SPLIT_SEED)

    train_loader, val_loader, class_names = (
        create_dataloaders(args.train_dir)
    )

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Classes:", class_names)

    model = build_model(device)

    model, best_val_accuracy = train_model(
        model,
        train_loader,
        val_loader,
        device
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "ConvNeXt-Small",
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "validation_accuracy": best_val_accuracy,
            "seed": SEED
        },
        args.output
    )

    print("\n======================================")
    print("TRAINING COMPLETE")
    print("======================================")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )
    print("Checkpoint saved to:", args.output)


if __name__ == "__main__":
    main()