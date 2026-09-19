import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, top_k_accuracy_score


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes=37):
    """返回 Top-1、Top-5、Macro-F1、全部标签与预测"""
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = all_probs.argmax(axis=1)

    top1 = (all_preds == all_labels).mean()
    top5 = top_k_accuracy_score(all_labels, all_probs, k=5, labels=np.arange(num_classes))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "top1": top1,
        "top5": top5,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "preds": all_preds,
    }


def plot_confusion_matrix(labels, preds, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds, labels=np.arange(len(class_names)))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix (37 classes)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"混淆矩阵已保存到 {save_path}")

    np.fill_diagonal(cm_norm, 0)
    pairs = []
    for i in range(len(class_names)):
        j = cm_norm[i].argmax()
        pairs.append((class_names[i], class_names[j], cm_norm[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("\n最易混淆的品种对（Top 5）：")
    for true_c, pred_c, rate in pairs[:5]:
        print(f"  {true_c}  →  {pred_c}  错误率 {rate:.2%}")
    return pairs[:5]