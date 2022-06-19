import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch import einsum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Infomax Loss =========================================
class GlobalDiscriminator(nn.Module):
    def __init__(self, sz):
        super(GlobalDiscriminator, self).__init__()
        self.l0 = nn.Linear(sz, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return self.l2(h)
    
class GlobalDiscriminatorConv(nn.Module):
    def __init__(self, sz):
        super(GlobalDiscriminatorConv, self).__init__()
        self.l0 = nn.Conv2d(sz, 128, kernel_size=1)
        self.l1 = nn.Conv2d(128, 128, kernel_size=1)
        self.l2 = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return self.l2(h)    
    
class PriorDiscriminator(nn.Module):
    def __init__(self, sz):
        super(PriorDiscriminator, self).__init__()
        self.l0 = nn.Linear(sz, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))    
        
class MILinearBlock(nn.Module):
    def __init__(self, feature_sz, units=2048, bln=True):
        super(MILinearBlock, self).__init__()
        # Pre-dot product encoder for "Encode and Dot" arch for 1D feature maps
        self.feature_nonlinear = nn.Sequential(
            nn.Linear(feature_sz, units, bias=False),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Linear(units, units),
        )
        self.feature_shortcut = nn.Linear(feature_sz, units)
        self.feature_block_ln = nn.LayerNorm(units)

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((units, feature_sz), dtype=np.bool)
        for i in range(feature_sz):
            eye_mask[i, i] = 1

        self.feature_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.feature_shortcut.weight.data.masked_fill_(
            torch.tensor(eye_mask), 1.0)
        self.bln = bln

    def forward(self, feat):
        f = self.feature_nonlinear(feat) + self.feature_shortcut(feat)
        if self.bln:
            f = self.feature_block_ln(f)

        return f
    
class GlobalDiscriminatorDot(nn.Module):
    def __init__(self, sz, units=2048, bln=True):
        super(GlobalDiscriminatorDot, self).__init__()
        self.block_a = MILinearBlock(sz, units=units, bln=bln)
        self.block_b = MILinearBlock(sz, units=units, bln=bln)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        features1=None,
        features2=None,
    ):

        # Computer cross modal loss
        feat1 = self.block_a(features1)
        feat2 = self.block_b(features2)

        feat1, feat2 = map(lambda t: F.normalize(
            t, p=2, dim=-1), (feat1, feat2))

        # ## Method 1
        # # Dot product and sum
        # o = torch.sum(feat1 * feat2, dim=1) * self.temperature.exp()

        # ## Method 2
        # o = self.cos(feat1, feat2) * self.temperature.exp()

        # Method 3
        o = einsum("n d, n d -> n", feat1, feat2) * self.temperature.exp()

        return o    
    
class DeepInfoMaxLoss(nn.Module):
    def __init__(self, type="concat"):
        super().__init__()
        if type=="concat":
            self.global_d = GlobalDiscriminator(sz=64+64)
        elif type=="dot":
            self.global_d = GlobalDiscriminatorDot(sz=64)
        else:
            self.global_d = GlobalDiscriminatorConv(sz=64+64)

    def forward(self, proto_label_pos, proto_label_neg, proto_unlabel_pos):
        Ej = -F.softplus(-self.global_d(proto_unlabel_pos, proto_label_pos)).mean()
        Em = F.softplus(self.global_d(proto_unlabel_pos, proto_label_neg)).mean()
        LOSS = (Em - Ej)
        
        return LOSS