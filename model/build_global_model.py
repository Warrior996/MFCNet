# encoding: utf-8
import re
import logging
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence
from model.convnext import convnext_base
from model.resnet import resnet50
from model.densenet import densenet121
from model.pyconv_densenet import pyconv_densenet121
from model.res2net_v1 import res2net50_v1b
from model.resnet_acmix import ACmix_ResNet
from model.densenet_acmix import densenet121_ACmix
from model.resnet_acmix3 import ACmix3_ResNet
from model.densenet_acmix3 import densenet121_ACmix3
from main import arg_parse
args = arg_parse()

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, pretrained, num_classes):
        super(DenseNet121, self).__init__()
        self.densenet121 = densenet121(pretrained=pretrained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.densenet121(x)
        return x

class DenseNet121_ACmix(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(DenseNet121_ACmix, self).__init__()
        self.densenet121_ACmix = densenet121_ACmix(pretrained=pretrained)
        num_ftrs = self.densenet121_ACmix.classifier.in_features
        self.densenet121_ACmix.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.densenet121_ACmix(x)
        return x

class DenseNet121_ACmix3(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(DenseNet121_ACmix3, self).__init__()
        self.densenet121_ACmix = densenet121_ACmix3(pretrained=pretrained)
        num_ftrs = self.densenet121_ACmix.classifier.in_features
        self.densenet121_ACmix.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.densenet121_ACmix(x)
        return x

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)

from functools import partial
norm_layer = partial(LayerNorm2d, eps=1e-6)

class Convnext_base(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(Convnext_base, self).__init__()
        self.convnext_base = convnext_base(pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.Sequential(norm_layer(1024),
                                  nn.Flatten(1))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.convnext_base.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x


class Vision_transformer(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(Vision_transformer, self).__init__()
        self.vision_transformer = vit_b_32(pretrained = pretrained)
        self.vision_transformer.heads = nn.Sequential(
            nn.Linear(768, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vision_transformer(x)
        return x

class ResNet50(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(pretrained=pretrained)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.resnet50(x)
        return x

class Res2Net50_v1b(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(Res2Net50_v1b, self).__init__()
        self.res2net50_v1b = res2net50_v1b(pretrained=pretrained)
        num_ftrs = self.res2net50_v1b.fc.in_features
        self.res2net50_v1b.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.res2net50_v1b(x)
        return x

class ResNet50_ACmix(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(ResNet50_ACmix, self).__init__()
        self.resnet50_ACmix = ACmix_ResNet(pretrained=pretrained)
        num_ftrs = self.resnet50_ACmix.fc.in_features
        self.resnet50_ACmix.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.resnet50_ACmix(x)
        return x


class ResNet50_ACmix3(nn.Module):

    def __init__(self, pretrained, num_classes):
        super(ResNet50_ACmix3, self).__init__()
        self.resnet50_ACmix = ACmix3_ResNet(pretrained=pretrained)
        num_ftrs = self.resnet50_ACmix.fc.in_features
        self.resnet50_ACmix.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.resnet50_ACmix(x)
        return x

def build_model():

    # initialize and load the model
    if args.model == 'densenet':
        Global_Branch_model = DenseNet121(pretrained=args.pretrained, num_classes=args.num_class).cuda()
    elif args.model == 'densenet_ACmix':
        Global_Branch_model = DenseNet121_ACmix(pretrained=args.pretrained, num_classes=args.num_class).cuda()
    elif args.model == 'densenet_ACmix3':
        Global_Branch_model = DenseNet121_ACmix3(pretrained=args.pretrained, num_classes=args.num_class).cuda()
    elif args.model == 'Res2Net50':
        Global_Branch_model = Res2Net50_v1b(pretrained=args.pretrained, num_classes=args.num_class).cuda()
    elif args.model == 'ResNet50':
        Global_Branch_model = ResNet50(pretrained=args.pretrained, num_classes=args.num_class).cuda()
    elif args.model == 'ResNet50_ACmix':
        Global_Branch_model = ResNet50_ACmix(pretrained=args.pretrained, num_classes=args.num_class).cuda()
    elif args.model == 'ResNet50_ACmix3':
        Global_Branch_model = ResNet50_ACmix3(pretrained=args.pretrained, num_classes=args.num_class).cuda()
    elif args.model == 'ConvNeXt_Base':
        Global_Branch_model = Convnext_base(pretrained=args.pretrained, num_classes=args.num_class).cuda()



    return Global_Branch_model


if __name__ == '__main__':
    model = Convnext_base(pretrained=True, num_classes=14).cuda()
    print(model)