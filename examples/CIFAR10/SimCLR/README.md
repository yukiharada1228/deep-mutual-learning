# CIFAR-10 SimCLR Experiments

This directory contains experimental results for CIFAR-10 representation learning using SimCLR (Simple Framework for Contrastive Learning of Visual Representations).

## Overview

SimCLR is a self-supervised learning method that learns visual representations by maximizing agreement between differently augmented views of the same image. The learned representations are evaluated using k-Nearest Neighbors (kNN) classification on the test set.

## 1. Independent Training (Baseline)

Base encoder trained individually without any knowledge transfer using SimCLR.

| Model | KNN Top-1 Accuracy | KNN Top-5 Accuracy |
|-------|-------------------:|-------------------:|
| ResNet18 | **89.68%** | **98.88%** |
| ResNet50 | **91.15%** | **99.09%** |

## 2. SimCLR with DisCO (Distillation with Contrastive Learning)

Knowledge distillation from ResNet50 (teacher) to ResNet18 (student) using contrastive learning.

| Student Model | Teacher Model | KNN Top-1 Accuracy | KNN Top-5 Accuracy | Improvement vs Baseline |
|---------------|---------------|-------------------:|-------------------:|------------------------:|
| ResNet18 | ResNet50 | **90.09%** | **98.82%** | **+0.41%** |

The DisCO approach improves ResNet18's performance from 89.68% (independent) to 90.09% by learning from the ResNet50 teacher model, demonstrating effective knowledge transfer through contrastive distillation.
