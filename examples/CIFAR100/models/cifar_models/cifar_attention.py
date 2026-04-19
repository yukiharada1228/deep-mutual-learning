from collections import namedtuple

import torch
import torch.nn as nn

from .cifar_resnet import CifarResNet, ResNetBasicblock


AttentionBranchOutput = namedtuple(
    "AttentionBranchOutput", ["aux_logits", "logits", "attention"]
)


class ConvBnDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBnDownsample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class CifarResNetAttentionWrapper(nn.Module):
    def __init__(self, backbone, num_classes=None, return_attention_output=False):
        super(CifarResNetAttentionWrapper, self).__init__()
        if not isinstance(backbone, CifarResNet):
            raise TypeError("backbone must be an instance of CifarResNet")

        if not isinstance(backbone.layer3[0], ResNetBasicblock):
            raise TypeError(
                "Only CifarResNet backbones with ResNetBasicblock are supported"
            )

        self.backbone = backbone
        self.return_attention_output = return_attention_output
        backbone_num_classes = backbone.fc.out_features
        if num_classes is not None and num_classes != backbone_num_classes:
            raise ValueError("num_classes must match backbone.fc.out_features")
        self.num_classes = backbone_num_classes

        layer3_blocks = len(backbone.layer3)
        layer2_channels = backbone.layer2[-1].bn_b.num_features
        layer3_channels = backbone.layer3[-1].bn_b.num_features

        self.att_layer3 = self._make_attention_layer(
            inplanes=layer2_channels,
            planes=layer3_channels,
            blocks=layer3_blocks,
        )
        self.bn_att = nn.BatchNorm2d(layer3_channels)
        self.att_conv = nn.Conv2d(
            layer3_channels,
            self.num_classes,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.bn_att2 = nn.BatchNorm2d(self.num_classes)
        self.att_conv2 = nn.Conv2d(
            self.num_classes,
            self.num_classes,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.att_conv3 = nn.Conv2d(
            self.num_classes, 1, kernel_size=3, padding=1, bias=False
        )
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self._init_attention_branch()

    def _make_attention_layer(self, inplanes, planes, blocks):
        layers = [
            ResNetBasicblock(
                inplanes,
                planes,
                stride=1,
                downsample=ConvBnDownsample(inplanes, planes, stride=1),
            )
        ]
        for _ in range(1, blocks):
            layers.append(ResNetBasicblock(planes, planes))
        return nn.Sequential(*layers)

    def _init_attention_branch(self):
        modules = [
            self.att_layer3,
            self.bn_att,
            self.att_conv,
            self.bn_att2,
            self.att_conv2,
            self.att_conv3,
            self.bn_att3,
        ]
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, (2.0 / n) ** 0.5)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _forward_impl(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        att_features = self.bn_att(self.att_layer3(x))
        att_logits = self.relu(self.bn_att2(self.att_conv(att_features)))
        attention = self.sigmoid(self.bn_att3(self.att_conv3(att_logits)))

        aux_logits = self.att_conv2(att_logits)
        aux_logits = self.att_gap(aux_logits)
        aux_logits = torch.flatten(aux_logits, 1)

        refined = x * attention + x
        refined = self.backbone.layer3(refined)
        refined = self.backbone.avgpool(refined)
        refined = torch.flatten(refined, 1)
        logits = self.backbone.fc(refined)

        return AttentionBranchOutput(
            aux_logits=aux_logits, logits=logits, attention=attention
        )

    def forward_with_attention(self, x):
        return self._forward_impl(x)

    def forward(self, x):
        output = self._forward_impl(x)
        if self.return_attention_output:
            return output
        return output.logits


def wrap_with_attention(backbone, return_attention_output=False):
    return CifarResNetAttentionWrapper(
        backbone, return_attention_output=return_attention_output
    )


def resnet20_abn(num_classes=10, return_attention_output=False):
    from .cifar_resnet import resnet20

    return CifarResNetAttentionWrapper(
        resnet20(num_classes=num_classes),
        return_attention_output=return_attention_output,
    )


def resnet32_abn(num_classes=10, return_attention_output=False):
    from .cifar_resnet import resnet32

    return CifarResNetAttentionWrapper(
        resnet32(num_classes=num_classes),
        return_attention_output=return_attention_output,
    )


def resnet44_abn(num_classes=10, return_attention_output=False):
    from .cifar_resnet import resnet44

    return CifarResNetAttentionWrapper(
        resnet44(num_classes=num_classes),
        return_attention_output=return_attention_output,
    )


def resnet56_abn(num_classes=10, return_attention_output=False):
    from .cifar_resnet import resnet56

    return CifarResNetAttentionWrapper(
        resnet56(num_classes=num_classes),
        return_attention_output=return_attention_output,
    )


def resnet110_abn(num_classes=10, return_attention_output=False):
    from .cifar_resnet import resnet110

    return CifarResNetAttentionWrapper(
        resnet110(num_classes=num_classes),
        return_attention_output=return_attention_output,
    )


def resnet1202_abn(num_classes=10, return_attention_output=False):
    from .cifar_resnet import resnet1202

    return CifarResNetAttentionWrapper(
        resnet1202(num_classes=num_classes),
        return_attention_output=return_attention_output,
    )
