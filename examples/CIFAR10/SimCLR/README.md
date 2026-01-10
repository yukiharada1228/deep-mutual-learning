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

## 2. SimCLR with Deep Mutual Learning (DML)

Coming soon.

## 3. SimCLR with DisCO (Distillation with Contrastive Learning)

Coming soon.
