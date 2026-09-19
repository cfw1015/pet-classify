import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from dataset import get_dataloaders
from utils.metrics import evaluate_model, plot_confusion_matrix
from utils.gradcam import GradCAM, visualize_gradcam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--tag", type=str, default="baseline")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.tag}] 使用设备：{device}")
    print(f"[{args.tag}] 加载权重：{args.ckpt}")

    _, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    dataset = val_loader.dataset.dataset
    class_names = dataset.classes if hasattr(dataset, "classes") else [str(i) for i in range(37)]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 37)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    # ===== 1. 指标 =====
    metrics = evaluate_model(model, test_loader, device)
    print(f"\n[{args.tag}] 测试集 Top-1={metrics['top1']:.4f} | "
          f"Top-5={metrics['top5']:.4f} | Macro-F1={metrics['macro_f1']:.4f}")

    # ===== 2. 混淆矩阵 =====
    plot_confusion_matrix(metrics["labels"], metrics["preds"], class_names,
                          save_path=f"confusion_matrix_{args.tag}.png")

    # ===== 3. Grad-CAM（正确 + 错误各一张）=====
    gradcam = GradCAM(model, model.layer4[-1])

    correct_done, wrong_done = False, False
    for imgs, labels in test_loader:
        imgs_dev = imgs.to(device)
        with torch.no_grad():
            preds = model(imgs_dev).argmax(dim=1).cpu()

        for i in range(imgs.size(0)):
            if not correct_done and preds[i] == labels[i]:
                cam, _ = gradcam.generate(imgs_dev[i:i+1], labels[i].item())
                visualize_gradcam(model, imgs[i], cam,
                                  save_path=f"gradcam_correct_{args.tag}.png",
                                  title=f"Correct: {class_names[labels[i]]}")
                correct_done = True
            elif not wrong_done and preds[i] != labels[i]:
                cam, _ = gradcam.generate(imgs_dev[i:i+1], preds[i].item())
                visualize_gradcam(model, imgs[i], cam,
                                  save_path=f"gradcam_wrong_{args.tag}.png",
                                  title=f"Wrong: True={class_names[labels[i]]} Pred={class_names[preds[i]]}")
                wrong_done = True
        if correct_done and wrong_done:
            break

    print(f"\n[{args.tag}] 评估完成 ✅")


if __name__ == "__main__":
    main()