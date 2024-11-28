import sys
sys.path.append("/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE")

from network.autoencoder.vae import VAE
from network.pointnet_encoder import PointNetEncoder
from typing import List
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, args, obj_inchannel=4,
                 cvae_encoder_sizes=[1024, 512, 256],
                 cvae_decoder_sizes=[256, 256, 61]):
        super(Autoencoder, self).__init__()
        self.args = args
        self.obj_inchannel = obj_inchannel
        self.cvae_encoder_sizes = cvae_encoder_sizes
        self.cvae_decoder_sizes = cvae_decoder_sizes

        self.hand_encoder = PointNetEncoder(
            global_feat=True, feature_transform=True, channel=3)

        self.autoencoder = VAE(encoder_layer_sizes=self.cvae_encoder_sizes,
                               decoder_layer_sizes=self.cvae_decoder_sizes)

        # self.linear_layer = nn.Linear(2048, 1024) #蒸馏
    def encoder(self, x):
        z = self.autoencoder.encoder(x)
        return z

    def decoder(self, z):
        x = self.autoencoder.decoder(z)
        return x

    def forward(self, hand):
        '''
        :param hand_param: [B, 778 , 3]
        :return: reconstructed hand vertex
        '''

        hand_glb_feature, _, _ = self.hand_encoder(hand)  # [B, 1024]

        z = self.encoder(hand_glb_feature)

        x = self.decoder(z)

        return x


if __name__ == '__main__':
    model = Autoencoder(args=None, obj_inchannel=4,
                        cvae_encoder_sizes=[1024, 512, 256],
                        cvae_decoder_sizes=[256, 256, 61])
    hand_param = torch.randn(3, 3, 778, )
    print('params {}M'.format(sum(p.numel()
          for p in model.parameters()) / 1000000.0))
    model.eval()
    x = model(hand_param)
    print(x.size())
