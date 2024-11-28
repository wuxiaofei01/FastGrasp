from network.fuse_encoder import FuseEncoder
from network.language_encoder import LanguageNetEncoder
from network.autoencoder.autoencoder import Autoencoder
from network.pointnet_encoder import PointNetEncoder
import torch
from torch.nn import Module
from typing import Dict, Tuple
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from network.diffusion.pointnet2.pointnet2_ssg_sem import PointNet2SemSegSSG
import json
import sys
sys.path.append("/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE")


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDIM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T +
                                            1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


class DDIM(nn.Module):
    def __init__(self,args , nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDIM, self).__init__()
        self.nn_model = nn_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v.to(device))

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss().to(device)
        self.loss_l1 = nn.L1Loss(reduction='mean').to(device)

        self.obj_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=4).to(device)
        self.language_encoder = LanguageNetEncoder(args).to(device)

        self.fuse_encoder = FuseEncoder()

    def forward(self, x, obj_feature=None, language_feature=None):
        '''
        x : [B , 256 , 3]
        '''
        if obj_feature is not None:
            obj_feature, _, _ = self.obj_encoder(obj_feature)
            # mask = torch.rand_like(obj_feature) > 0.5 # drop out 50% of the features
            # obj_feature = obj_feature * mask

        if language_feature is not None:
            language_feature = self.language_encoder(language_feature)

        if obj_feature is not None and language_feature is not None:
            c = self.fuse_encoder(obj_feature, language_feature)
        elif obj_feature is not None and language_feature is None:
            c = obj_feature
        elif obj_feature is None and language_feature is not None:
            c = language_feature
        elif obj_feature is None and language_feature is None:
            c = None
        # t ~ Uniform(0, n_T)
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x).to(self.device)  # eps ~ N(0, 1)
        x_t = (
            self.sqrtab[_ts, None,  None] * x
            + self.sqrtmab[_ts, None,  None] * noise.to(self.device)
        ) 
        
        predict_noise = self.nn_model(x_t.to(self.device), _ts / self.n_T, c)
        predict_x = (x_t-self.sqrtmab[_ts, None,  None] * predict_noise)/self.sqrtab[_ts, None,  None]

        return self.loss_mse(noise, predict_noise) , predict_x
    def sample(self, n_sample, size, device, guide_w=0.0, obj_feature=None, language_feature=None):
        x_i = torch.randn(n_sample, *size).to(device)
        if obj_feature is not None:
            obj_feature, _, _ = self.obj_encoder(obj_feature)
        if language_feature is not None:
            language_feature = self.language_encoder(language_feature)

        if obj_feature is not None and language_feature is not None:
            c = self.fuse_encoder(obj_feature, language_feature)
        elif obj_feature is not None:
            c = obj_feature
        elif language_feature is not None:
            c = language_feature
        else:
            c = None
        if c is not None:
            # c = c.repeat(2, 1)
            c = torch.cat([c, torch.zeros_like(c)], dim=0)

        for i in range(self.n_T, 0, -1):
            
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample)

            # double batch
            x_i = x_i.repeat(2, 1, 1)
            t_is = t_is.repeat(2)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            # split predictions and compute weighting

            eps = self.nn_model(x_i.to(self.device),
                                t_is.to(self.device), c)

            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i

    def sample_ddim(self,
                    n_sample,
                    size,
                    device,
                    guide_w=2,
                    obj_feature=None,
                    language_feature=None,
                    simple_var=True,
                    ddim_step=10,
                    eta=1):
        x = torch.randn(n_sample, *size).to(device)
        
        if obj_feature is not None:
            obj_feature, _, _ = self.obj_encoder(obj_feature)
        if language_feature is not None:
            language_feature = self.language_encoder(language_feature)

        if obj_feature is not None and language_feature is not None:
            c = self.fuse_encoder(obj_feature, language_feature)
        elif obj_feature is not None:
            c = obj_feature
        elif language_feature is not None:
            c = language_feature
        else:
            c = None
        '''classifier free'''
        if c is not None:
            c = torch.cat([c, torch.zeros_like(c)], dim=0)            

        if simple_var:
            eta = 1
        ts = torch.linspace(self.n_T, 0,(ddim_step + 1)).to(device).to(torch.long)

        batch_size = x.shape[0]

        for i in range(1, ddim_step + 1):
            #先不考虑classfy free
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample)

            x = x.repeat(2, 1, 1)
            t_is = t_is.repeat(2)

            eps = self.nn_model(x.to(self.device),t_is.to(self.device), c)

            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2


            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            ab_cur = self.alphabar_t[cur_t]
            ab_prev = self.alphabar_t[prev_t] if prev_t >= 0 else 1

            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)

            '''classifier free'''
            x = x[:n_sample,:]

            noise = torch.randn_like(x)
            first_term = (ab_prev / ab_cur)**0.5 * x
            second_term = ((1 - ab_prev - var)**0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise
            else:
                third_term = var**0.5 * noise
            x = first_term + second_term + third_term

        return x
    def sample_Gaussian(self, n_sample, size, device, guide_w=0.0, obj_feature=None, language_feature=None):
        x_i = torch.randn(n_sample, *size).to(device)
        if obj_feature is not None:
            obj_feature, _, _ = self.obj_encoder(obj_feature)
        if language_feature is not None:
            language_feature = self.language_encoder(language_feature)

        if obj_feature is not None and language_feature is not None:
            c = self.fuse_encoder(obj_feature, language_feature)
        elif obj_feature is not None:
            c = obj_feature
        elif language_feature is not None:
            c = language_feature
        else:
            c = None
        if c is not None:
            # c = c.repeat(2, 1)
            c = torch.cat([c, torch.randn_like(c)], dim=0)
        
        for i in range(self.n_T, 0, -1):
            
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample)

            # double batch
            x_i = x_i.repeat(2, 1, 1)
            t_is = t_is.repeat(2)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            # split predictions and compute weighting

            eps = self.nn_model(x_i.to(self.device),
                                t_is.to(self.device), c)

            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i

if __name__ == "__main__":
    with open("/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE/config/diffusion.json", 'r') as f:
        param = json.load(f)
    ddpm = DDIM(nn_model=PointNet2SemSegSSG(param).to("cuda"), betas=(
        1e-4, 0.02), n_T=1000, device="cuda", drop_prob=0.1)

    print("yes")
