"""
Step 2: Train the recognition CNN from scratch on your captured dataset.

    python 02_train_model.py --epochs 30

Loads dataset/<person_name>/*.png, trains a small CNN with random init
(no pretrained weights anywhere), and saves:
  - checkpoints/face_cnn.pt   (model weights)
  - checkpoints/classes.json  (index -> person name mapping)

Because "from scratch" datasets are usually small (a few hundred images
per person), aggressive data augmentation is what keeps the model from
memorizing the exact 250 images you fed it instead of learning the face.
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from model import FaceCNN


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.35, contrast=0.35),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full_dataset = FaceDataset(args.data_dir, transform=None)
    if len(full_dataset.classes) < 2:
        raise RuntimeError(
            "Need at least 2 people captured in dataset/ before training "
            "(the model needs multiple classes to distinguish between)."
        )
    print(f"Found {len(full_dataset.classes)} classes: {full_dataset.classes}")
    print(f"Total images: {len(full_dataset)}")

    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Apply different transforms to train vs val via wrapper
    class TransformWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            img = Image.open(path).convert("L")
            img = self.transform(img)
            return img, label

    train_ds = TransformWrapper(train_subset, train_transform)
    val_ds = TransformWrapper(val_subset, val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FaceCNN(num_classes=len(full_dataset.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, cosine = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (cosine.argmax(dim=1) == labels).sum().item()
            train_total += imgs.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, cosine = model(imgs)
                val_correct += (cosine.argmax(dim=1) == labels).sum().item()
                val_total += imgs.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step(val_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss {train_loss/train_total:.4f} | "
              f"train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            model.eval()

            # Compute centroids from the FULL dataset with clean (no augmentation)
            # transforms so they match exactly what the live camera produces at
            # inference time. Using all images gives more stable centroid estimates.
            all_clean_ds = FaceDataset(args.data_dir, transform=val_transform)
            all_clean_loader = DataLoader(all_clean_ds, batch_size=args.batch_size, shuffle=False)

            centroids = torch.zeros(len(full_dataset.classes), 128, device=device)
            counts = torch.zeros(len(full_dataset.classes), device=device)
            with torch.no_grad():
                for imgs, labels in all_clean_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    embeds = model.extract_features(imgs)
                    for embed, label in zip(embeds, labels):
                        centroids[label] += embed
                        counts[label] += 1
            centroids = torch.nn.functional.normalize(centroids / counts.unsqueeze(1).clamp(min=1), p=2, dim=1)

            torch.save({
                "model_state": model.state_dict(),
                "centroids": centroids.cpu(),
            }, os.path.join(args.out_dir, "face_cnn.pt"))

    with open(os.path.join(args.out_dir, "classes.json"), "w") as f:
        json.dump(full_dataset.classes, f)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Saved model & centroids to {args.out_dir}/face_cnn.pt")
    print(f"Saved class mapping to {args.out_dir}/classes.json")


if __name__ == "__main__":
    main()