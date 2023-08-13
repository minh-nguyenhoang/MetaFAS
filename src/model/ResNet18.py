import torch
from torch import nn
from torchvision import models

class FeatureExtractor(nn.Module):
    
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        base_model = models.resnet18(pretrained=pretrained)
        self.nets = nn.Sequential(*(list(base_model.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dep_est = nn.Sequential(nn.Upsample((32,32)),
                                     nn.Conv2d(512, 128, 3,1,1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace= True),
                                     nn.Conv2d(128,1,3,1,1),
                                     nn.ReLU(inplace= True))
        
    def forward(self, x):
        conv_feat = self.nets(x)
        feat = self.avgpool(conv_feat).squeeze(3).squeeze(2)
        map_pr = self.dep_est(conv_feat).squeeze(1)
        return feat, map_pr