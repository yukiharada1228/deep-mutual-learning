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
