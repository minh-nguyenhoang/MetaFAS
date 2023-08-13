import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import math

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None


revgrad = GradientReversal.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha = .1):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
    
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding= 0, dilation= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, **kwarg) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

class CDConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7, trainable = False, **kwargs):

        super(CDConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias=False) 
        if trainable:
            self.theta = nn.Parameter(torch.tensor(theta), requires_grad= True)
        else:
            self.register_buffer('theta', torch.tensor(theta))

    def forward(self, x):
        out_normal = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            # [C_out,C_in, kernel_size,kernel_size] = self.weight.shape
            # kernel_diff = self.weight.sum((2,3), keepdim= True)
            out_diff = F.conv2d(input=x, weight=self.weight.sum((2,3), keepdim= True), bias=self.bias, stride=self.stride, padding=0, groups=self.groups)

            return out_normal - self.theta * out_diff


class LDConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2, **kwargs):
        super(LDConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        
#         self.theta = nn.Parameter(torch.tensor(theta), requires_grad= False)
        self.register_buffer('center_mask',torch.tensor([  [0, 0, 0],
                                                        [0, 1, 0],
                                                        [0, 0, 0]]))
        self.register_buffer('base_mask', torch.ones(self.weight.size()))
#         self.center_mask = nn.Parameter(torch.tensor([  [0, 0, 0],
#                                                         [0, 1, 0],
#                                                         [0, 0, 0]]), requires_grad= False)
#         self.base_mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones([self.weight.size(0),self.weight.size(1)]), requires_grad=True)
        self.learnable_theta = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
    def forward(self, x):

        # Reference: `Searching Central Difference Convolutional Networks for Face Anti-Spoofing` (CVPR'20)
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.weight.sum(2).sum(2) [:, :, None, None]
        
        out_diff = F.conv2d(input=x, weight=self.weight * mask, bias=self.bias, stride=self.stride,
                            padding=self.padding,
                            dilation= self.dilation,
                            groups=self.groups)
        
        return out_diff
    

def ldconv2d(input, weight, center_mask, base_mask, learnable_mask,
             learnable_theta, bias, stride, padding, dilation, groups):

    mask = base_mask - learnable_theta * learnable_mask[:, :, None, None] * \
        center_mask * weight.sum(2).sum(2) [:, :, None, None]
    
    out_diff = F.conv2d(input=input, weight= weight * mask, bias=bias, stride=stride,
                            padding=padding,
                            dilation= dilation,
                            groups=groups)
        
    return out_diff

def cdconv2d(input, weight, bias, theta, stride, padding, dilation, groups):
    out_normal = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    if math.fabs(theta.cpu().item() - 0.0) < 1e-8:
        return out_normal 
    else:
        # [C_out,C_in, kernel_size,kernel_size] = self.weight.shape
        # kernel_diff = self.weight.sum((2,3), keepdim= True)
        out_diff = F.conv2d(input=input, weight=weight.sum((2,3), keepdim= True), bias=bias, stride=stride, padding=0, groups=groups)

        return out_normal - theta * out_diff

