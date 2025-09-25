import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange


# 当使用 padding_mode='reflect' 时，PyTorch 要求：
# padding 必须小于输入尺寸的每个维度
# 对于 2×2 的输入，最大 padding 只能是 1
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)
        

#     def forward(self, x):
#         x_avg = torch.mean(x, dim=1, keepdim=True)
#         x_max, _ = torch.max(x, dim=1, keepdim=True)
#         x2 = torch.concat([x_avg, x_max], dim=1)
#         sattn = self.sa(x2)
#         return sattn

#   自适应池化
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 先自适应池化到固定大小，再卷积
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # 如果输入太小，先上采样
        if H < 7 or W < 7:
            x_pooled = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
        else:
            x_pooled = x
            
        x_avg = torch.mean(x_pooled, dim=1, keepdim=True)
        x_max, _ = torch.max(x_pooled, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        
        attn = self.sa(x2)
        
        # 恢复到原始尺寸
        if H < 7 or W < 7:
            attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
            
        return attn
    

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

    
# class PixelAttention(nn.Module):
#     def __init__(self, dim):
#         super(PixelAttention, self).__init__()
#         self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, pattn1):
#         B, C, H, W = x.shape
#         x = x.unsqueeze(dim=2) # B, C, 1, H, W
#         pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
#         x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
#         x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
#         pattn2 = self.pa2(x2)
#         pattn2 = self.sigmoid(pattn2)
#         return pattn2

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        # 小尺寸用小卷积核
        self.conv_small = nn.Conv2d(2 * dim, dim, 3, padding=1, 
                                   padding_mode='reflect', groups=dim, bias=True)
        # 大尺寸用大卷积核
        self.conv_large = nn.Conv2d(2 * dim, dim, 7, padding=3, 
                                   padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        # 简单的自适应策略
        if min(H, W) <= 6:
            pattn2 = self.conv_small(x2)
        else:
            pattn2 = self.conv_large(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
  
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

if __name__ == '__main__':
    input1=torch.randn(50,512,3,3)
    input2=torch.randn(50,512,3,3)
    fusion = CGAFusion(512)
    output=fusion(input1, input2)
