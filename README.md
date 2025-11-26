<p style="display: inline">
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
  <img src="https://img.shields.io/badge/-Pytorch-11b3d3.svg?logo=pytorch&style=for-the-badge">
  <img src="https://img.shields.io/badge/-arxiv-B31B1B.svg?logo=arxiv&style=for-the-badge">
  <!-- <img src="https://img.shields.io/badge/-Docker-eb7739.svg?logo=docker&style=for-the-badge"> -->
</p>

# KnowledgeTransferGraph

This repository implements the "Knowledge Transfer Graph for Deep Collaborative Learning" described in the [ACCV 2020 accepted paper](https://openaccess.thecvf.com/content/ACCV2020/html/Minami_Knowledge_Transfer_Graph_for_Deep_Collaborative_Learning_ACCV_2020_paper.html). Notably, this implementation deviates from the original paper in certain aspects related to hyperparameter tuning.

## Replace CorrectGate with NegativeLinearGate
The original CorrectGate was replaced by NegativeLinearGate in the implementation. This change was made to specifically enhance the model's performance in controlling knowledge transfer along the temporal dimension, with the aim of improving overall accuracy.

## Methodology

### Overview of Knowledge Transfer Graph (KTG)

Knowledge Transfer Graph is a framework for collaborative learning among multiple deep learning models. Each model is represented as a node in the graph, and knowledge transfer between models is controlled by edges. This architecture enables models with different architectures to share knowledge with each other, leading to improved performance for individual models.

### Graph Structure

KTG consists of two main components:

1. **Node**: Each node represents a single deep learning model. Each node has its own independent optimizer, scheduler, and evaluation metrics.

2. **Edge**: Connections that control knowledge transfer between nodes. Each edge consists of a combination of a loss function and a gate function.

### Edge Mechanism

Each edge consists of two components:

- **Loss Function (Criterion)**: Calculates the loss for knowledge transfer
  - **Self-edge**: Computes the loss between the model's own predictions and ground truth labels (typically uses `CrossEntropyLoss`)
  - **Cross-edge**: Computes the knowledge transfer loss between models (uses `KLDivLoss` to calculate temperature-scaled KL divergence)

- **Gate Function**: Dynamically adjusts the loss weight according to the epoch, controlling knowledge transfer along the temporal dimension

### Gate Function Types

This implementation provides four types of gate functions:

1. **ThroughGate**: Passes the loss through unchanged for all epochs (weight = 1.0)

2. **CutoffGate**: Completely disables gradient contribution from the loss (weight = 0.0)

3. **PositiveLinearGate**: Applies a linearly increasing weight according to the epoch number (initial: 0.0 → final: 1.0)
   - Strategy: Suppress knowledge transfer in early stages and strengthen it in later stages

4. **NegativeLinearGate**: Applies a linearly decreasing weight according to the epoch number (initial: 1.0 → final: 0.0)
   - Strategy: Emphasize knowledge transfer in early stages and transition to individual learning in later stages
   - Implemented as a replacement for the original `CorrectGate`, achieving improved performance in controlling knowledge transfer along the temporal dimension

### Collaborative Learning Mechanism

The learning process for each batch proceeds as follows:

1. **Forward Pass**: All nodes (models) perform forward propagation on the same input data and generate outputs

2. **Loss Calculation**: For each node, the following losses are calculated:
   - Loss from self-edge: Loss between the model's own predictions and ground truth labels
   - Loss from other models: Knowledge transfer losses with outputs from other models
   - Total loss = Σ(losses from all edges)

3. **Backward Pass**: Each node independently performs backpropagation and updates its own parameters

4. **Knowledge Transfer**: Knowledge is transferred between models through losses controlled by gate functions

### Loss Function Details

#### KLDivLoss (for Knowledge Transfer)

For knowledge transfer between models, we use temperature-scaled KL divergence:

- The temperature parameter `T` adjusts the distribution of soft targets
- Multiplying by `T²` as a gradient scale correction compensates for gradient shrinkage due to temperature scaling
- This enables effective knowledge transfer between different architectures

### Implementation Features

- **Flexible Graph Configuration**: Can set any number of nodes and edges
- **Independent Optimization**: Each node has its own independent optimizer and scheduler
- **Mixed Precision Training**: Supports automatic mixed precision training with `torch.amp.autocast` and `GradScaler`
- **Hyperparameter Optimization**: Integration with Optuna enables optimization of gate function selection and model architectures

### Usage Example

For the CIFAR-100 dataset, the following configuration is possible:

- Place three different models (ResNet32, ResNet110, WideResNet28-2) as nodes
- Set different gate functions for edges between each node
- Explore optimal gate function combinations through Optuna hyperparameter optimization

## Usage
To use the Knowledge Transfer Graph in your project, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/yukiharada1228/KnowledgeTransferGraph.git
cd KnowledgeTransferGraph
```
2. Install the package:
```bash
uv sync
```

## Acknowledgements

This implementation is based on the original paper ["Knowledge Transfer Graph for Deep Collaborative Learning"](https://arxiv.org/abs/1909.04286) by Soma Minami, Tsubasa Hirakawa, Takayoshi Yamashita, and Hironobu Fujiyoshi. I acknowledge and appreciate their valuable contributions to the field.
