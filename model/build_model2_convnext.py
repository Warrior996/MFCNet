# encoding: utf-8
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from collections import OrderedDict
from model.convnext import convnext_base


class Global_Branch(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, pretrained, num_classes):
        super(Global_Branch, self).__init__()
        self.convnext_base = convnext_base(pretrained=pretrained)
        self.convnext_base.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.convnext_base(x)
        features = self.convnext_base.features(x)
        out = F.relu(features, inplace=True)
        out_after_pooling = F.avg_pool2d(out, kernel_size=7, stride=1).view(out.size(0), -1)
        # out_after_pooling = self.convnext_base.avgpool(features)
        out = self.convnext_base.classifier(out_after_pooling)
        return out, features, out_after_pooling

class Local_Branch(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Local_Branch, self).__init__()
        self.convnext_base = convnext_base(pretrained=pretrained)
        self.convnext_base.classifier = nn.Sequential(nn.Linear(1024, num_classes), nn.Sigmoid())
        
    def forward(self, x, mask):
        #x: N*C*W*H
        features = self.convnext_base.features(x)
        sigmoid_x = torch.sigmoid(features)

        # mask = self.adapool(mask)
        mask = F.adaptive_avg_pool2d(mask, (7, 7))
        mask = mask.ge(0.5).float() # 0,1 binarization
        x = torch.mul(sigmoid_x, mask)

        out_after_pooling = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)
        out = self.convnext_base.classifier(out_after_pooling)
        return out, features, out_after_pooling
        
    # 绘制热图
    # def forward(self, x):
    #     x = self.densenet121(x)
    #     return x
    
class Fusion_Branch(nn.Module):
    def __init__(self, input_size, output_size):
        super(Fusion_Branch, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, global_pool, local_pool):
        #fusion = torch.cat((global_pool.unsqueeze(2), local_pool.unsqueeze(2)), 2).cuda()
        #fusion = fusion.max(2)[0]#.squeeze(2).cuda()
        #print(fusion.shape)
        fusion = torch.cat((global_pool,local_pool), 1).cuda()
        fusion_var = torch.autograd.Variable(fusion)
        x = self.fc(fusion_var)
        x = self.Sigmoid(x)
        return x




