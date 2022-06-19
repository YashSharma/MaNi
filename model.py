import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else: 
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, domain=False, proto=False, projection=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.domain_classifier = nn.Sequential(
            RevGrad(),
            OutConv(64, 1)
        )
        
        if projection:
            self.proto_projection = nn.Sequential(
                OutConv(64, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:
            self.proto_projection = nn.Sequential(
                nn.Identity()
            )
                                 
        self.proto_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        self.domain = domain
        self.proto = proto

    def forward_one(self, x, x_map=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        domain_pred = None
        proto_pred_pos = None
        proto_pred_neg = None        
        
        if self.domain:
            domain_pred = self.domain_classifier(x)
            
        if self.proto:
            # Generate Mask
            if x_map == None:
                logits_map_pos = torch.round(torch.sigmoid(logits))
                logits_map_neg = 1.-logits_map_pos
            else:                
                logits_map_pos = x_map
                logits_map_neg = 1.-logits_map_pos
                
            x_map_pos = self.proto_pool(self.proto_projection(x)*logits_map_pos)
            x_map_neg = self.proto_pool(self.proto_projection(x)*logits_map_neg)
            
            normalizing_factor = np.prod(np.array(logits_map_pos.shape[-2:]))
            
            proto_pred_pos = x_map_pos*((normalizing_factor)/(logits_map_pos.sum(list(range(1, logits_map_pos.ndim))).reshape((logits_map_pos.size(0), 1))+1e-6))
            proto_pred_neg = x_map_neg*((normalizing_factor)/(logits_map_neg.sum(list(range(1, logits_map_pos.ndim))).reshape((logits_map_pos.size(0), 1))+1e-6))
        return logits, domain_pred, proto_pred_pos, proto_pred_neg
    
    def forward(self, x_label, x_unlabel=None, x_label_map=None, x_unlabel_map=None, validation=False):
        if validation:
            logit_label, domain_label,  proto_label_pos, proto_label_neg = self.forward_one(x_label)    
            return logit_label, domain_label, proto_label_pos, proto_label_neg            
        
        if x_unlabel == None:
            return self.forward_one(x_label)
        
        logit_label, domain_label, proto_label_pos, proto_label_neg = self.forward_one(x_label, x_label_map)
        logit_unlabel, domain_unlabel, proto_unlabel_pos, proto_unlabel_neg = self.forward_one(x_unlabel, x_unlabel_map)
        
        return logit_label, domain_label, proto_label_pos, proto_label_neg, \
                logit_unlabel, domain_unlabel, proto_unlabel_pos, proto_unlabel_neg