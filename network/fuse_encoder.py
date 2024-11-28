import torch
import torch.nn as nn
import numpy as np
import math


class FuseEncoder(nn.Module):
    def __init__(self, input_dim=2048, output_dim=1024):
        super(FuseEncoder, self).__init__()

        self.linear_layer = nn.Linear(input_dim, output_dim).cuda()

    def forward(self, obj_language , language):
        fuse_feature = torch.cat((obj_language, language), dim=1) 

        fuse_feature = self.linear_layer(fuse_feature)

        return fuse_feature