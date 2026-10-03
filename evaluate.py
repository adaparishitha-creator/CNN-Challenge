import argparse
import os
import subprocess
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


NUM_CLASSES = 16
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EXPECTED_CLASSES = [
    "Bedroom",
    "Coast",
    "Flower",
    "Forest",
    "Highway",
    "Industrial",
    "InsideCity",
    "Kitchen",
    "LivingRoom",
    "Mountain",
    "Office",
    "OpenCountry",
    "Store",
    "Street",
    "Suburb",
    "TallBuilding",
]


def prepare_convnext():
    repo_dir = "ConvNeXt"

    if not os.path.isdir(repo_dir):
        print("Downloading official ConvNeXt repository...")

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/facebookresearch/ConvNeXt.git",
                repo_dir,
            ],
            check=True,
        )

    repo_path = os.path.abspath(repo_dir)

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from models.convnext import convnext_small
    except Exception as exc:
        raise RuntimeError(
            "Could not import ConvNeXt-Small from the official "
            "ConvNeXt repository."
        ) from exc

    return convnext_small


def build_model(checkpoint_path, device):
    convnext_small = prepare_convnext()

    # Build the same ConvNeXt-Small architecture used in Experiment 6.
    # Pretrained ImageNet-22K weights are NOT needed here because the
    # submitted checkpoint already contains the fine-tuned weights.
    model = convnext_small(
        pretrained=False,
        in_22k=True,
        num_classes=21841,
    )

    model.head = nn.Linear(
        model.head.in_features,
        NUM_CLASSES,
    )

    print("Loading checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def create_test_loader(test_dir):
    test_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=test_transform,
    )

    if len(test_dataset) != 400:
        raise ValueError(
            f"Expected 400 test images, "
            f"but found {len(test_dataset)}."
        )

    if len(test_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but found {len(test_dataset.classes)}."
        )

    if test_dataset.classes != EXPECTED_CLASSES:
        raise ValueError(
            "Test class ordering does not match "
            "the training class ordering.\n"
            f"Found: {test_dataset.classes}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader, test_dataset


@torch.inference_mode()
def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    test_loss = running_loss / total
    test_accuracy = correct / total

    return test_loss, test_accuracy


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the selected Experiment 6 "
            "ConvNeXt-Small checkpoint."
        )
    )

    parser.add_argument(
        "--test-dir",
        type=str,
        default="test",
        help="Path to the test dataset.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiment6_best_model.pth",
        help="Path to the Experiment 6 checkpoint.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.test_dir):
        raise FileNotFoundError(
            f"Test directory not found: {args.test_dir}"
        )

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("====================================")
    print("CNN CHALLENGE - FINAL EVALUATION")
    print("====================================")
    print("Device:", device)

    test_loader, test_dataset = (
        create_test_loader(args.test_dir)
    )

    print("Test images:", len(test_dataset))
    print("Number of classes:", len(test_dataset.classes))
    print("Classes:", test_dataset.classes)

    model, checkpoint = build_model(
        args.checkpoint,
        device,
    )

    print("Checkpoint loaded successfully.")

    if isinstance(checkpoint, dict):
        if "validation_accuracy" in checkpoint:
            print(
                "Saved validation accuracy:",
                f"{checkpoint['validation_accuracy'] * 100:.2f}%"
            )

        if "test_accuracy" in checkpoint:
            print(
                "Saved test accuracy:",
                f"{checkpoint['test_accuracy'] * 100:.2f}%"
            )

    print("\nEvaluating 400 test images...")

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print("\n====================================")
    print("FINAL RESULTS")
    print("====================================")
    print(f"Test Loss: {test_loss:.4f}")
    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()