import os
import time
import torch
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset.oakink_dataset import Oakink
from dataset.grab_dataset import Grab
from network.autoencoder.autoencoder import Autoencoder
from network.diffusion.ddim import DDIM
from network.diffusion.pointnet2.pointnet2_ssg_sem import PointNet2SemSegSSG
import numpy as np
import random
from utils import utils_loss
from utils.utils import  makepath, makelogger , convert_euler_to_rotmat
from utils.loss import CVAE_loss_mano, CMap_loss, CMap_loss1, CMap_loss3, CMap_loss4, inter_penetr_loss, CMap_consistency_loss, set_random_seed
from pytorch3d.loss import chamfer_distance
import mano
import ipdb
import sys
import json
from torch.utils.data.sampler import SubsetRandomSampler
from manotorch.manolayer import ManoLayer, MANOOutput
from torch.utils.tensorboard import SummaryWriter
from evaluation.vis import vis_hand
from manopth.manolayer import grabManoLayer
from tqdm import tqdm
from torch.utils.data import Subset
from datetime import datetime
import shutil

grab_mano_layer = grabManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("assets/mano_v1_2/models"), use_pca=False, joint_rot_mode="rotmat").to("cuda")


def train(args,writer , epoch,autoencoder, model, train_loader, device, optimizer, logger, checkpoint_root ,best_train_loss, rh_mano, rh_faces):
    logs = defaultdict(list)
    model.train()
    autoencoder.eval()
    autoencoder.requires_grad_(False)

    for batch_idx, input in enumerate(train_loader):
        obj_pc = input["obj_pc"].to(device)
        hand_param = input["hand_param"].to(device)
        if args.dataset =="oakink":
            gt_mano = rh_mano(hand_param[:, 10:58] , hand_param[:,:10])
            hand_xyz = gt_mano.verts.to(device) +  hand_param[:,None, 58:] # [B,778,3]
        else:
            th_v_template = input["th_v_template"].to(device)
            hand_xyz = input["hand_verts"].to(device)

        optimizer.zero_grad()
        '''encoder'''
        hand_feature, _, _ = autoencoder.hand_encoder(hand_xyz.permute(0,2,1))
        # pointnet
        z = autoencoder.encoder(hand_feature).view(hand_feature.shape[0] , -1,3) # [B , 256 , 3] <- [B, 768]
        # unet
        diff_loss,predict_x = model(z , obj_feature = obj_pc , language_feature = None)


        recon_param = autoencoder.decoder(predict_x.reshape(predict_x.shape[0], -1).to(device))


        if args.mano == "oakink":
            recon_mano = rh_mano(
                recon_param[:, 10:58], recon_param[:, :10])
            recon_xyz = recon_mano.verts.to(
                device) + recon_param[:, None, 58:]  # [B,778,3]
        else:
            handbetas = hand_param[:, :10 ]
            handeuler = recon_param[:, 10:58].view(-1,16,3)  # [ B , 16 , 3 ]
            handrot = convert_euler_to_rotmat(handeuler)# [ B , 16 , 3 , 3]
            th_trans =recon_param[:, 58:] # [ B , 3]
            hand_verts_list = []
            for k , x in enumerate(handrot):
                hand_vert, _ ,_= grab_mano_layer(handrot[k].unsqueeze(0).to(device),th_betas=handbetas[k].unsqueeze(0).to(device), th_trans=th_trans[k].unsqueeze(0).to(device), th_v_template=th_v_template[k].unsqueeze(0))
                hand_verts_list.append(hand_vert)
            recon_xyz = torch.stack(hand_verts_list).squeeze(1)


        # mano param loss
        param_loss = torch.nn.functional.mse_loss(
            recon_param, hand_param, reduction='none').sum() / recon_param.size(0)

        loss = diff_loss
        logs['loss'].append(loss.item())
        logs['diff_loss'].append(diff_loss.item())
        logs['param_loss'].append(param_loss.item())      

        loss.backward()
        optimizer.step()
        

    mean_loss = sum(logs['loss']) / len(logs['loss'])
    mean_diff_loss = sum(logs['diff_loss']) / len(logs['diff_loss'])
    mean_param_loss = sum(logs['param_loss']) / len(logs['param_loss'])

    writer.add_scalar('loss', mean_loss , epoch)
    writer.add_scalar('diff_loss', mean_diff_loss , epoch)
    writer.add_scalar('param_loss', mean_param_loss, epoch)



    out_str = "Epoch: {:02d}/{:02d},{:7s},Loss {:9.5f},Diff_loss {:9.5f} , param Loss {:9.5f},,Best Loss: {:9.5f}".format(
        epoch, args.epochs,"train",
        mean_loss,
        mean_diff_loss,
        mean_param_loss,
        min(mean_loss , best_train_loss)
    )
    logger(out_str)

    if mean_loss < best_train_loss :
        save_name = os.path.join(checkpoint_root, 'model_best_train_diffusion.pth')
        torch.save({
            'network': model.state_dict(),
            'epoch': epoch
        }, save_name)

        save_name = os.path.join(checkpoint_root, 'model_best_train_ae.pth')
        torch.save({
            'network': autoencoder.state_dict(),
            'epoch': epoch
        }, save_name)

    return min(mean_loss , best_train_loss)

def val(args,writer , epoch,autoencoder, model, train_loader, device, optimizer, logger, checkpoint_root ,best_loss, rh_mano, rh_faces, mode='val'):
   # validation
    logs = defaultdict(list)
    autoencoder.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, input in enumerate(train_loader):
            obj_pc = input["obj_pc"].to(device)
            hand_param = input["hand_param"].to(device)
            if args.dataset =="oakink":
                gt_mano = rh_mano(hand_param[:, 10:58] , hand_param[:,:10])
                hand_xyz = gt_mano.verts.to(device) +  hand_param[:,None, 58:] # [B,778,3]
            else:
                th_v_template = input["th_v_template"].to(device)
                hand_xyz = input["hand_verts"].to(device)

            hand_feature, _, _ = autoencoder.hand_encoder(hand_xyz.permute(0,2,1))

            z = autoencoder.encoder(hand_feature).view(hand_feature.shape[0] , -1,3) # [B , 256 , 3] <- [B, 768]

            diff_loss,predict_x = model(z , obj_feature = obj_pc , language_feature = None)

            recon_param = autoencoder.decoder(predict_x.reshape(predict_x.shape[0], -1).to(device))


            if args.mano == "oakink":
                recon_mano = rh_mano(
                    recon_param[:, 10:58], recon_param[:, :10])
                recon_xyz = recon_mano.verts.to(
                    device) + recon_param[:, None, 58:]  # [B,778,3]
            else:
                handbetas = hand_param[:, :10 ]
                handeuler = recon_param[:, 10:58].view(-1,16,3)  # [ B , 16 , 3 ]
                handrot = convert_euler_to_rotmat(handeuler)# [ B , 16 , 3 , 3]
                th_trans =recon_param[:, 58:] # [ B , 3]
                hand_verts_list = []
                for k , x in enumerate(handrot):
                    hand_vert, _ ,_= grab_mano_layer(handrot[k].unsqueeze(0).to(device),th_betas=handbetas[k].unsqueeze(0).to(device), th_trans=th_trans[k].unsqueeze(0).to(device), th_v_template=th_v_template[k].unsqueeze(0))
                    hand_verts_list.append(hand_vert)
                recon_xyz = torch.stack(hand_verts_list).squeeze(1)
                


            param_loss = torch.nn.functional.mse_loss(
                recon_param, hand_param, reduction='none').sum() / recon_param.size(0)


            loss = diff_loss


            logs['loss'].append(loss.item())
            logs['diff_loss'].append(diff_loss.item())
            logs['param_loss'].append(param_loss.item()) 

    mean_loss = sum(logs['loss']) / len(logs['loss'])
    mean_diff_loss= sum(logs['diff_loss']) / len(logs['diff_loss'])
    mean_param_loss = sum(logs['param_loss']) / len(logs['param_loss'])


    writer.add_scalar('loss', mean_loss , epoch)
    writer.add_scalar('diff_loss', mean_diff_loss, epoch)
    writer.add_scalar('param_loss', mean_param_loss, epoch)

    out_str = "Epoch: {:02d}/{:02d},{:7s},Loss {:9.5f},Diff Loss {:9.5f}, param Loss {:9.5f},Best Loss: {:9.5f}"\
        .format(epoch, args.epochs, mode, mean_loss, mean_diff_loss, mean_param_loss, min(best_loss, mean_loss))


    logger(out_str)

    if mean_loss < best_loss:
        save_name = os.path.join(checkpoint_root, 'model_best_{}_diffusion.pth'.format(mode))
        torch.save({
            'network': model.state_dict(),
            'epoch': epoch
        }, save_name)

        save_name = os.path.join(checkpoint_root, 'model_best_{}_ae.pth'.format(mode))
        torch.save({
            'network': autoencoder.state_dict(),
            'epoch': epoch
        }, save_name)



    return min(best_loss, mean_loss)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,default = "config/oakink/diffusion.json",help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config, 'r') as configfile:
        config = json.load(configfile)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()

    # log file
    save_root = os.path.join('./logs', f"diffusion/{args.dataset}/{args.file_name}" )
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    log_root = save_root + '/exp.log'

    # logger
    logger = makelogger(makepath(os.path.join(log_root), isfile=True)).info
    starttime = datetime.now().replace(microsecond=0)
    logger('Started training %s' % (starttime))
    gpu_brand = torch.cuda.get_device_name(0) if args.use_cuda else None
    if args.use_cuda and torch.cuda.is_available():
        logger('Using 1 CUDA cores [%s] for training!' % (gpu_brand))
    logger(args)

    shutil.copy(args.config, save_root)


    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_num = 1
    """loar autoencoder model"""
    autoencoder = Autoencoder(
        args = args,
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_decoder_sizes=args.decoder_layer_sizes)
    checkpoint = torch.load(args.autoencoder_path, map_location=torch.device(device))
    new_state_dict = {}
    for key, value in checkpoint['network'].items():
        if key.startswith('module.'):
            new_key = key[7:]  
        else:
            new_key = key
        new_state_dict[new_key] = value
    autoencoder.load_state_dict(new_state_dict)
    autoencoder = autoencoder.to(device)
    """load diffusion model"""
    with open('config/diffusion.json', 'r') as f:
        diffusion_param = json.load(f)
        net = PointNet2SemSegSSG(diffusion_param).to(device)

    diffusion_model = DDIM(args=args,nn_model=net,
     betas=(1e-4, 0.02), n_T=1000, device=device, drop_prob=0.1)

    # multi-gpu
    # if device == torch.device("cuda"):
    #     torch.backends.cudnn.benchmark = True
    #     device_ids = range(torch.cuda.device_count())
    #     print("using {} cuda".format(len(device_ids)))
    #     if len(device_ids) > 1:
    #         diffusion_model = torch.nn.DataParallel(diffusion_model)

    #         device_num = len(device_ids)
    # dataset

    if 'Train' in args.train_mode:
        if args.dataset == "oakink":
            train_dataset = Oakink(mode="train", batch_size=args.batch_size, args=args)
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)
        else:
            train_dataset = Grab(mode="train", batch_size=args.batch_size, args=args , aug_ratio = args.aug_ratio)
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)

    if 'Val' in args.train_mode:
        if args.dataset == "oakink":
            val_dataset = Oakink(mode="val", batch_size=args.batch_size, args=args)
            val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.dataloader_workers)
        else:
            val_dataset = Grab(
                mode="val", batch_size=args.batch_size, args=args)
            val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.dataloader_workers)
    

    logger(len(train_loader))    
    logger(len(val_loader))

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.1, 0.2, 0.3, 0.4 , 0.5]], gamma=0.5)

    #writer
    train_writer = SummaryWriter(os.path.join(os.path.dirname(log_root), 'tensorboard/train') )
    val_writer = SummaryWriter(os.path.join(os.path.dirname(log_root), 'tensorboard/val') )
    test_writer = SummaryWriter(os.path.join(os.path.dirname(log_root), 'tensorboard/test') )
    
    # mano hand model
    with torch.no_grad():
        rh_mano = ManoLayer(center_idx=0, mano_assets_root="assets/mano_v1_2").to(device)
    rh_faces = rh_mano.th_faces.view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes
    rh_faces = rh_faces.repeat(args.batch_size, 1, 1).to(device) # [N, 1538, 3]
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        if 'Train' in args.train_mode:
            best_train_loss = train(args,train_writer, epoch, autoencoder,diffusion_model, train_loader, device, optimizer, logger,save_root, best_train_loss ,rh_mano, rh_faces,)
            scheduler.step()
        if 'Val' in args.train_mode:
            best_val_loss = val(args,val_writer, epoch, autoencoder,diffusion_model, val_loader, device, optimizer, logger,save_root, best_val_loss ,rh_mano, rh_faces,"val")
