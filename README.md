# 基于深度学习的牛津宠物细粒度图像分类

科研项目组本科生入组考核 · 3 天极速挑战

作者：XXX | 学号：XXX | 日期：2026-09

## 项目简介

本项目基于 PyTorch，在 Oxford-IIIT Pet（37 类细粒度宠物分类，约 7.3k 张）数据集上，
使用 ResNet-18 预训练骨干进行迁移学习，完成：

1. Baseline 训练：基础数据增强（RandomCrop + RandomHorizontalFlip）
2. 消融实验 A：Label Smoothing CrossEntropy（smoothing=0.1）
3. 消融实验 B：MixUp 数据增强（alpha=0.2）
4. 模型分析：混淆矩阵 + Grad-CAM 可视化 + Bad Case 分析

## 环境依赖

- Python 3.10
- PyTorch 2.x（CPU 或 CUDA）
- 其他依赖见 requirements.txt

安装：

    pip install -r requirements.txt

## 数据准备

数据会自动下载到 ./data/：

    python -c "from torchvision.datasets import OxfordIIITPet; OxfordIIITPet(root='./data', download=True)"

按 70% : 15% : 15% 进行分层抽样（Stratified Split），固定随机种子 seed=42。

## 一键复现

三组训练：

    python train.py --tag baseline
    python train.py --tag label_smooth --label_smoothing 0.1
    python train.py --tag mixup --use_mixup

三组评估：

    python evaluate.py --ckpt best_model_baseline.pth --tag baseline
    python evaluate.py --ckpt best_model_label_smooth.pth --tag label_smooth
    python evaluate.py --ckpt best_model_mixup.pth --tag mixup

查看训练曲线：

    tensorboard --logdir runs --port 6006

## 实验结果

| 实验配置 | Top-1 | Top-5 | Macro-F1 | ValAcc | 训练耗时 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | 0.9058 | 0.9946 | 0.9049 | 0.9384 | 20.79 min |
| + Label Smoothing 0.1 | 0.9130 | 0.9855 | 0.9121 | 0.9185 | 21.33 min |
| + MixUp (alpha=0.2) | 0.9167 | 0.9891 | 0.9161 | 0.9221 | 21.18 min |

结论：
- MixUp 测试集 Top-1 最高（0.9167），Macro-F1 也最高（0.9161），泛化最好
- Label Smoothing 次之（0.9130），相比 Baseline 提升 +0.72%
- Baseline ValAcc 虚高（0.9384），TestAcc 最低（0.9058）——典型过拟合

## 项目结构

    pet_classify/
    ├── data/
    ├── runs/
    ├── images/
    ├── utils/
    │   ├── metrics.py
    │   └── gradcam.py
    ├── dataset.py
    ├── train.py
    ├── evaluate.py
    ├── requirements.txt
    └── README.md

## 参考

- Oxford-IIIT Pet: https://www.robots.ox.ac.uk/~vgg/data/pets/
- ResNet: He et al., CVPR 2016
- Grad-CAM: Selvaraju et al., ICCV 2017
- MixUp: Zhang et al., ICLR 2018
