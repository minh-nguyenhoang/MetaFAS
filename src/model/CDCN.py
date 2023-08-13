import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np
from ..utils.layers import CDConv2d, LDConv2d, ldconv2d, cdconv2d
from ..utils.loss import KSubArcFace



class SpatialAttention(nn.Module):
    def __init__(self, kernel = 3):
        super(SpatialAttention, self).__init__()


        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x)


class CDCN(nn.Module):

    def __init__(self,in_channels = 3, basic_conv=CDConv2d, theta=0.7):   
        super(CDCN, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            basic_conv(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # self.featconv = nn.Conv2d(128*3, 512, 1,1,0)
        
        # self.lastconv1 = nn.Sequential(
        #     basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),    
        # )
        
        # self.lastconv2 = nn.Sequential(
        #     basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),    
        # )
        
        # self.lastconv3 = nn.Sequential(
        #     basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.ReLU(),    
        # )
        
        
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]  
        
        x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	  
        x_Block2_32x32 = self.downsample32x32(x_Block2)   # x [128, 32, 32]  
        
        x_Block3 = self.Block3(x_Block2)	    # x [128, 32, 32]  	
        x_Block3_32x32 = self.downsample32x32(x_Block3)   # x [128, 32, 32]  
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]  
        
        # x_feature = self.featconv(x_concat)
        #pdb.set_trace()
        
        # x = self.lastconv1(x_concat)    # x [128, 32, 32] 
        # x = self.lastconv2(x)    # x [64, 32, 32] 
        # x = self.lastconv3(x)    # x [1, 32, 32] 
        
        # map_x = x.squeeze(1)
        
        return x_concat


class CDCNpp(nn.Module):
    def __init__(self, in_channels = 3, basic_conv=CDConv2d, theta=0.7):   
        super(CDCNpp, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            basic_conv(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
            
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            
            basic_conv(128, int(128*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.6)),
            nn.ReLU(),  
            basic_conv(int(128*1.6), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            basic_conv(128, int(128*1.4), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.4)),
            nn.ReLU(),  
            basic_conv(int(128*1.4), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Original

        # self.featconv = nn.Conv2d(128*3, 512,1,1,0)
        
        # self.lastconv1 = nn.Sequential(
        #     basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     basic_conv(128, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.ReLU(),    
        # )
        
      
        self.sa1 = SpatialAttention(kernel = 7)
        self.sa2 = SpatialAttention(kernel = 5)
        self.sa3 = SpatialAttention(kernel = 3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   
        
        x_Block1 = self.Block1(x)	    	    	
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)   
        
        x_Block2 = self.Block2(x_Block1)	    
        attention2 = self.sa2(x_Block2)  
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)  
        
        x_Block3 = self.Block3(x_Block2)	    
        attention3 = self.sa3(x_Block3)  
        x_Block3_SA = attention3 * x_Block3	
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)   
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)  

        # x_feature = self.featconv(x_concat)
        
        # #pdb.set_trace()
        
        # map_x = self.lastconv1(x_concat)
        
        # map_x = map_x.squeeze(1)
        
        return x_concat
		
class DepthEstor(nn.Module):
    def __init__(self, basic_conv = LDConv2d, theta = .7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
    
    def forward(self, x_concat):
        return self.net(x_concat).squeeze(1)
  
class FeatureExtractor(nn.Module):
    
    def __init__(self, in_channels = 6, basic_conv = LDConv2d, theta = 0.3):
        super(FeatureExtractor, self).__init__()
        self.nets = CDCN(in_channels = in_channels,basic_conv= basic_conv, theta= theta)
        
    def forward(self, x):
        x_concat = self.nets(x)  
        return x_concat
    
class Classifier(nn.Module):
    def __init__(self, num_cls, basic_conv = LDConv2d, theta = 0.3, **kwargs) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride= 2),
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 2, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(2*16*16, num_cls)
        )
#         self.arcspoof = KSubArcFace(2*16*16, num_cls, k = 3,s = kwargs.get('s', 30), m= kwargs.get('m', .5))

    def forward(self, x, label: torch.Tensor|None = None, params: dict|None = None):
        if params is None:
            feat_flat = self.net(x)
            
            cls_out = feat_flat
#             cls_out = self.arcspoof(feat_flat, label)

        else: 
            device = x.device
            out = F.max_pool2d(x,  self.net[0].kernel_size,
                                   self.net[0].stride,
                                   self.net[0].padding,
                                   self.net[0].dilation,
                                   self.net[0].ceil_mode,
                                   self.net[0].return_indices)
            
            if str(type(self.net[1])).strip("'>").split('.')[-1] == 'LDConv2d':
                out = ldconv2d(out,
                            params['net.1.weight'].to(device),
                            params['net.1.center_mask'].to(device),
                            params['net.1.base_mask'].to(device),
                            params['net.1.learnable_mask'].to(device),
                            params['net.1.learnable_theta'].to(device),
                            params.get('net.1.bias'),
                            stride= self.net[1].stride,
                            padding= self.net[1].padding,
                            dilation= self.net[1].dilation,
                            groups= self.net[1].groups)
            elif str(type(self.net[1])).strip("'>").split('.')[-1] == 'Conv2d':
                out = F.conv2d(out,
                            params['net.1.weight'].to(device),
                            params.get('net.1.bias'),
                            stride= self.net[1].stride,
                            padding= self.net[1].padding,
                            dilation= self.net[1].dilation,
                            groups= self.net[1].groups)
            else:
                out = cdconv2d(out,
                            params['net.1.weight'].to(device),
                            params.get('net.1.bias'),
                            theta= params['net.1.theta'].to(device),
                            stride= self.net[1].stride,
                            padding= self.net[1].padding,
                            dilation= self.net[1].dilation,
                            groups= self.net[1].groups)
            
            out = F.batch_norm(out,
                               params['net.2.running_mean'].to(device),
                               params['net.2.running_var'].to(device),
                               params['net.2.weight'].to(device),
                               params['net.2.bias'].to(device),
                               training= self.net[2].training,
                               momentum= self.net[2].momentum,
                               eps = self.net[2].eps)
            
            out = F.relu(out, inplace= self.net[3].inplace)
            
            if str(type(self.net[4])).strip("'>").split('.')[-1] == 'LDConv2d':
                out = ldconv2d(out,
                            params['net.4.weight'].to(device),
                            params['net.4.center_mask'].to(device),
                            params['net.4.base_mask'].to(device),
                            params['net.4.learnable_mask'].to(device),
                            params['net.4.learnable_theta'].to(device),
                            params.get('net.4.bias'),
                            stride= self.net[4].stride,
                            padding= self.net[4].padding,
                            dilation= self.net[4].dilation,
                            groups= self.net[4].groups)
            elif str(type(self.net[4])).strip("'>").split('.')[-1] == 'Conv2d':
                out = F.conv2d(out,
                            params['net.4.weight'].to(device),
                            params.get('net.4.bias'),
                            stride= self.net[4].stride,
                            padding= self.net[4].padding,
                            dilation= self.net[4].dilation,
                            groups= self.net[4].groups)
            else:
                out = cdconv2d(out,
                            params['net.4.weight'].to(device),
                            params.get('net.4.bias'),
                            theta= params['net.4.theta'].to(device),
                            stride= self.net[4].stride,
                            padding= self.net[4].padding,
                            dilation= self.net[4].dilation,
                            groups= self.net[4].groups)
            
            out = F.batch_norm(out,
                               params['net.5.running_mean'].to(device),
                               params['net.5.running_var'].to(device),
                               params['net.5.weight'].to(device),
                               params['net.5.bias'].to(device),
                               training= self.net[5].training,
                               momentum= self.net[5].momentum,
                               eps = self.net[5].eps)
            
            out = F.relu(out, inplace= self.net[6].inplace)
            
            
            feat_flat = torch.flatten(out, start_dim= self.net[7].start_dim,
                                end_dim= self.net[7].end_dim)
            
            cls_out = F.linear(feat_flat, self.net[8].weight, self.net[8].bias)
            
#             cls_out = self.arcspoof(feat_flat, label, {'weight': params['arcspoof.weight'].to(device)})


        return cls_out
