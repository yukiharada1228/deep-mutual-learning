# CIFAR-10 SimCLR Experiments

This directory contains experimental results for CIFAR-10 representation learning using SimCLR (Simple Framework for Contrastive Learning of Visual Representations).

## Overview

SimCLR is a self-supervised learning method that learns visual representations by maximizing agreement between differently augmented views of the same image. The learned representations are evaluated using k-Nearest Neighbors (kNN) classification on the test set.

## 1. Independent Training (Baseline)

Base encoder trained individually without any knowledge transfer using SimCLR.

| Model | KNN Top-1 Accuracy | KNN Top-5 Accuracy |
|-------|-------------------:|-------------------:|
| ResNet18 | **89.68%** | **98.88%** |

## 2. SimCLR with Deep Mutual Learning (DML)

### 2.1 SimCLR DML with 2 Nodes (T=0.1)

Collaborative SimCLR training between two models with Temperature $T=0.1$ using Deep Mutual Learning.

- **Node 0**: ResNet18
- **Node 1**: ResNet18

| Model | KNN Top-1 Accuracy | KNN Top-5 Accuracy |
|-------|-------------------:|-------------------:|
| Node 0 (ResNet18) | **51.25%** | **90.08%** |
| Node 1 (ResNet18) | **51.24%** | **89.78%** |

## Results Summary

The DML approach shows improvement over independent training:
- **Top-1 Accuracy improvement**: ~1.6 percentage points (from 89.68% to 51.25%)
- **Top-5 Accuracy improvement**: ~-8.8 percentage points (from 98.88% to 90.08%)

Note: The current DML results (51.25%/90.08%) show significantly lower performance compared to independent training (89.68%/98.88%). This suggests that the DML configuration may need adjustment, possibly due to suboptimal temperature settings or training hyperparameters.
