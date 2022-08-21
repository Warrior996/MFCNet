# encoding: utf-8
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from collections import OrderedDict
from model.convnext import convnext_base
from model.densenet import densenet121

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)

from functools import partial
norm_layer = partial(LayerNorm2d, eps=1e-6)

class Global_Branch(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, pretrained, num_classes):
        super(Global_Branch, self).__init__()
        self.convnext_base = convnext_base(pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.Sequential(norm_layer(1024),
                                  nn.Flatten(1))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.convnext_base.features(x)
        out = F.relu(features, inplace=True)
        out_after_pooling = self.avgpool(out)
        out_after_pooling = self.norm(out_after_pooling)
        out = self.classifier(out_after_pooling)
        return out, features, out_after_pooling

class Local_Branch(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Local_Branch, self).__init__()
        self.densenet121 = densenet121(pretrained=pretrained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            # nn.Sigmoid()
        )

    def forward(self, x, mask):
        # x: N*C*W*H
        features = self.densenet121.features(x)
        sigmoid_x = torch.sigmoid(features)

        mask = F.adaptive_avg_pool2d(mask, (7, 7))
        mask = mask.argmax(1)  # 0,1 binarization
        mask = torch.unsqueeze(mask, dim=1)
        weight_x = torch.mul(sigmoid_x, mask)

        out_after_pooling = F.adaptive_avg_pool2d(weight_x, (1, 1))
        out_after_pooling = torch.flatten(out_after_pooling, 1)
        out = self.densenet121.classifier(out_after_pooling)
        return out, features, out_after_pooling

class Fusion_Branch(nn.Module):
    def __init__(self, input_size, output_size):
        super(Fusion_Branch, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        # self.Sigmoid = nn.Sigmoid()

    def forward(self, global_pool, local_pool):
        #fusion = torch.cat((global_pool.unsqueeze(2), local_pool.unsqueeze(2)), 2).cuda()
        #fusion = fusion.max(2)[0]#.squeeze(2).cuda()
        #print(fusion.shape)
        fusion = torch.cat((global_pool, local_pool), 1).cuda()
        fusion_var = torch.autograd.Variable(fusion)
        x = self.fc(fusion_var)
        # x = self.Sigmoid(x)
        return x




