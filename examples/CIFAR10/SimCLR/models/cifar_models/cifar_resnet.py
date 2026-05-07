import torch.nn as nn
import torchvision


def resnet50(num_classes=10):
    model = torchvision.models.resnet50(num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model
