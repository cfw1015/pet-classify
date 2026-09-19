# -*- coding: utf-8 -*-
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloaders


def mixup_data(x, y, alpha=0.2, device="cpu"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--use_randaugment", action="store_true")
    parser.add_argument("--tag", type=str, default="baseline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Tag: {args.tag}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, use_randaugment=args.use_randaugment
    )

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 37)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=f"runs/{args.tag}")
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if args.use_mixup:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, args.mixup_alpha, device)
                outputs = model(imgs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += torch.sum(pred == labels).item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / val_total

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        print(f"Epoch [{epoch+1}/{args.epochs}] | TrainLoss:{avg_train_loss:.4f} "
              f"| ValLoss:{avg_val_loss:.4f} | ValAcc:{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"best_model_{args.tag}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"[BEST] val_acc={best_val_acc:.4f} -> saved to {save_path}")

    elapsed = time.time() - start_time
    writer.close()
    print(f"\nTrain done. Best val_acc = {best_val_acc:.4f} | Time = {elapsed/60:.2f} min")

    model.load_state_dict(torch.load(f"best_model_{args.tag}.pth", map_location=device))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, pred = torch.max(out, dim=1)
            test_total += labels.size(0)
            test_correct += torch.sum(pred == labels).item()
    test_acc = test_correct / test_total
    print(f"\n[TEST] best model test_acc = {test_acc:.4f}")


if __name__ == "__main__":
    main()