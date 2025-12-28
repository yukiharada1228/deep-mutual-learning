<p style="display: inline">
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
  <img src="https://img.shields.io/badge/-Pytorch-11b3d3.svg?logo=pytorch&style=for-the-badge">
  <img src="https://img.shields.io/badge/-arxiv-B31B1B.svg?logo=arxiv&style=for-the-badge">
  <!-- <img src="https://img.shields.io/badge/-Docker-eb7739.svg?logo=docker&style=for-the-badge"> -->
</p>

# KnowledgeTransferGraph

This repository implements the "Knowledge Transfer Graph for Deep Collaborative Learning" described in the [ACCV 2020 accepted paper](https://openaccess.thecvf.com/content/ACCV2020/html/Minami_Knowledge_Transfer_Graph_for_Deep_Collaborative_Learning_ACCV_2020_paper.html). This implementation faithfully follows the original paper, including all four gate functions proposed in the paper.

## Gate Functions
The implementation includes the four gate functions as described in the original paper:
- **ThroughGate**: Passes the loss without modification
- **CutoffGate**: Blocks knowledge transfer by returning zero loss
- **LinearGate**: Linearly increases the weight of the loss from 0 to 1 as training progresses
- **CorrectGate**: Filters samples based on the correctness of teacher and student predictions, using only samples where the teacher made correct predictions

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
