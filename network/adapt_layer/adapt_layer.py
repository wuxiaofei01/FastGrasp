"""
the magnitude-preserving unet proposed in https://arxiv.org/abs/2312.02696 by Karras et al.
"""

import math
from math import sqrt, ceil
from functools import partial

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

# from einops import rearrange, repeat, pack, unpack
import sys
sys.path.append("/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE")
from network.pointnet_encoder import PointNetEncoder
# from network.adapt_layer.attention import Attend
# from attention import Attend
# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def append(arr, el):
    arr.append(el)


def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def divisible_by(numer, denom):
    return (numer % denom) == 0


def l2norm(t, dim = -1, eps = 1e-12):
    return F.normalize(t, dim = dim, eps = eps)

def interpolate_1d(x, length, mode = 'bilinear'):
    x = rearrange(x, 'b c t -> b c t 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    return rearrange(x, 'b c t 1 -> b c t')

# mp activations
# section 2.5

class MPSiLU(Module):
    def forward(self, x):
        return F.silu(x) / 0.596

# gain - layer scaling

class Gain(Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x * self.gain


class MPAdd(Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, res):
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den


# forced weight normed conv2d and linear
# algorithm 1 in paper

def normalize_weight(weight, eps = 1e-4):
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')

class Conv1d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        init_dirac = False,
        concat_ones_to_input = False   # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size)
        self.weight = nn.Parameter(weight)

        if init_dirac:
            nn.init.dirac_(self.weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)

        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 1, 0), value = 1.)

        return F.conv1d(x, weight, padding = 'same')

class Linear(Module):
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)

# mp fourier embeds

class MPFourierEmbedding(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * sqrt(2)


class AdaptLayer(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        attn_flash = False,
        upsample = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.upsample = upsample
        self.needs_skip = not upsample

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv1d(dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv1d(dim_out, dim_out, 3)
        )

        self.res_conv = Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        if has_attn:
            self.attn = Attention(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

    def forward(
        self,
        x,
        emb = None
    ):
        if self.upsample:
            x = interpolate_1d(x, x.shape[-1] * 2, mode = 'bilinear')

        res = self.res_conv(x)

        x = self.block1(x)
        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x



class AdaptLayer2(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size , intention = False):
        super().__init__()

        self.conditional = conditional
        self.intention = intention
        if self.conditional:
            input_size = latent_size + condition_size
            if intention == True:
                self.embedding = nn.Embedding(5, 1024).to('cuda')
            else:
                self.obj_encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=4).to("cuda")
        else:
            input_size = latent_size

        self.MLP = nn.Sequential().to("cuda")


        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                # self.MLP.add_module(name="BN{:d}".format(i), module=nn.BatchNorm1d(out_size))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
                # self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(p=0.5))

    def forward(self, z, obj=None):
        if self.conditional:
            if self.intention == False:
                c, _, _ = self.obj_encoder(obj)
                z = torch.cat((z, c), dim=-1)
            else:
                c = self.embedding(obj.long())
                z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)

        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_size, out_size)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x += residual
        return x

class ResNet(nn.Module):
    def __init__(self, in_size, out_size, num_blocks=3):
        super(ResNet, self).__init__()
        self.initial = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([ResidualBlock(out_size, out_size) for _ in range(num_blocks)])
        self.final = nn.Linear(out_size, out_size)

    def forward(self, x):
        x = self.initial(x)
        x = self.relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        return x

class AdaptLayer3(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, condition_size, resnet_blocks=3):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
            self.obj_encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=4)
        else:
            input_size = latent_size
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="R{:d}".format(i), module=ResNet(out_size, out_size, num_blocks=resnet_blocks))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z, c=None):
        if self.conditional and c is not None:
            c,_,_ = self.obj_encoder(c)
            z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class AdaptLayerTransformer(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
            self.obj_encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=4)
        else:
            input_size = latent_size

        # Embedding layer to project input to the Transformer dimension
        self.embedding = nn.Linear(input_size, layer_sizes[0])
        
        # Transformer encoder layer
        encoder_layers = TransformerEncoderLayer(d_model=layer_sizes[0], nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Output layer to project back to the desired output size
        self.fc = nn.Linear(layer_sizes[0], layer_sizes[-1])

    def forward(self, z, obj=None):
        if self.conditional and obj is not None:
            obj, _, _ = self.obj_encoder(obj)
            z = torch.cat((z, obj), dim=-1)
        
        # Add a dummy sequence dimension
        z = self.embedding(z).unsqueeze(1)  # Shape: (batch_size, seq_len=1, d_model)
        
        # Pass through the Transformer encoder
        z = self.transformer_encoder(z).squeeze(1)  # Shape: (batch_size, d_model)
        
        # Project to output size
        x = self.fc(z)

        return x

if __name__ == '__main__':

    '''AdaptLayer1'''
    # d = AdaptLayer(100)
    # x = torch.randn(100,768)
    # x = d(x)
    # print(x.shape)
    # print(1)
    '''AdaptLayer2'''
    # d = AdaptLayer2([768 , 1024 ,2048 , 512 , 61] , 768 , conditional= False ,condition_size= 1024)
    # x = torch.rand(128,768)
    # x = d(x)
    # print(x.shape)
    '''AdaptLayer3'''
    # layer_sizes = [32, 64, 128]
    # latent_size = 10
    # conditional = False
    # condition_size = 5
    # resnet_blocks = 2

    # x = torch.randn(1, latent_size)  # 输入大小为 1xlatent_size
    # model = AdaptLayer2(layer_sizes, latent_size, conditional, condition_size, resnet_blocks)
    # output = model(x)
    # print(output.shape)  # 应该输出 torch.Size([1, 128])
    d = AdaptLayerTransformer([768 , 1024 ,2048 , 512 , 61] , 768 , conditional= True ,condition_size= 1024)
    x = torch.rand(128,768)
    x = d(x , torch.rand(128,3 ,3000 ) )
    print(x.shape , )

