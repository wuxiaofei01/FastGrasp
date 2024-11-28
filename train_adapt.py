from manopth.manolayer import grabManoLayer
from evaluation.vis import vis_hand
from torch.utils.tensorboard import SummaryWriter
from manotorch.manolayer import ManoLayer, MANOOutput
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
from network.adapt_layer.adapt_layer import AdaptLayer, AdaptLayer2, AdaptLayer3 ,AdaptLayerTransformer
import numpy as np
import random
from utils import utils_loss
from utils.loss import CVAE_loss_mano, CMap_loss, CMap_loss1, CMap_loss3, CMap_loss4, inter_penetr_loss, CMap_consistency_loss, set_random_seed
from pytorch3d.loss import chamfer_distance
import mano
import ipdb
import sys
import json
from utils.utils import  makepath, makelogger , convert_euler_to_rotmat
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
from tqdm import tqdm
import shutil
local_time = time.localtime(time.time())

grab_mano_layer = grabManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join(
    "assets/mano_v1_2/models"), use_pca=False, joint_rot_mode="rotmat").to("cuda")


def train(args, writer, epoch, autoencoder, model, adapt_layer, train_loader, device, optimizer, logger, checkpoint_root, best_train_loss, rh_mano, rh_faces ,mode ="train"):
    logs = defaultdict(list)

    autoencoder.train()
    adapt_layer.train()
    model.train()

    a, b, c, d, e = args.weight
    for batch_idx, input in enumerate(train_loader):
        obj_pc = input["obj_pc"].to(device)
        hand_param = input["hand_param"].to(device)
        if args.dataset == "oakink":
            gt_mano = rh_mano(hand_param[:, 10:58], hand_param[:, :10])
            hand_xyz = gt_mano.verts.to(
                device) + hand_param[:, None, 58:]  # [B,778,3]
        else:
            th_v_template = input["th_v_template"].to(device)
            hand_xyz = input["hand_verts"].to(device)

        optimizer.zero_grad()
        '''encoder'''
        hand_feature, _, _ = autoencoder.hand_encoder(hand_xyz.permute(0, 2, 1))
        z = autoencoder.encoder(hand_feature).view(hand_feature.shape[0], -1, 3)  # [B , 256 , 3] <- [B, 768]
        diff_loss, predict_x = model(z, obj_feature=obj_pc, language_feature=None)

        """adapt_layer"""
        adapt_x = adapt_layer(predict_x.reshape(predict_x.shape[0], -1).to(device), obj_pc)
        '''decoder'''
        recon_param = autoencoder.decoder(predict_x.reshape(predict_x.shape[0], -1).to(device) + adapt_x)

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

        # obj xyz NN dist and idx
        obj_nn_dist_gt, obj_nn_idx_gt = utils_loss.get_NN(
            obj_pc.permute(0, 2, 1)[:, :, :3], hand_xyz)
        obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(
            obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz)


        param_loss = torch.nn.functional.mse_loss(recon_param, hand_param, reduction='none').sum() / recon_param.size(0)
        # mano recon xyz loss, KLD loss
        recon_loss_num, _ = chamfer_distance(
            recon_xyz, hand_xyz, point_reduction='sum', batch_reduction='mean')

        cmap_loss = CMap_loss3(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz, obj_nn_dist_recon < 0.01**2)
        # cmap consistency loss
        consistency_loss = CMap_consistency_loss(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz, hand_xyz,
                                                 obj_nn_dist_recon, obj_nn_dist_gt)
        # inter penetration loss
        rh_face = rh_faces[0]
        rh_faces = rh_face.expand(recon_xyz.shape[0], -1, -1)  # align the B
        penetr_loss = inter_penetr_loss(recon_xyz, rh_faces, obj_pc.permute(0, 2, 1)[:, :, :3],
                                        obj_nn_dist_recon, obj_nn_idx_recon)

        loss = a * recon_loss_num + b * param_loss + c * \
            cmap_loss + d * penetr_loss + e * consistency_loss
        loss.backward()
        optimizer.step()

        logs['diff_loss'].append(diff_loss)
        logs['recon_loss'].append(recon_loss_num)
        logs['loss'].append(loss.item())
        logs['param_loss'].append(param_loss.item())
        logs['cmap_loss'].append(cmap_loss.item())
        logs['penetr_loss'].append(penetr_loss.item())
        logs['cmap_consistency'].append(consistency_loss.item())
    writer.add_scalar('loss', sum(logs['loss']) / len(logs['loss']), epoch)
    writer.add_scalar('diff_loss', sum(
        logs['diff_loss']) / len(logs['diff_loss']), epoch)
    writer.add_scalar('recon_loss', sum(
        logs['recon_loss']) / len(logs['recon_loss']), epoch)
    writer.add_scalar('param_loss', sum(
        logs['param_loss']) / len(logs['param_loss']), epoch)
    writer.add_scalar('cmap_loss', sum(
        logs['cmap_loss']) / len(logs['cmap_loss']), epoch)
    writer.add_scalar('penetr_loss', sum(
        logs['penetr_loss']) / len(logs['penetr_loss']), epoch)
    writer.add_scalar('cmap_consistency', sum(
        logs['cmap_consistency']) / len(logs['cmap_consistency']), epoch)

    mean_recon_loss = sum(logs['penetr_loss']) / len(logs['penetr_loss'])
    
    best_train_loss = min(mean_recon_loss , best_train_loss)

    out_str = "Epoch: {:02d}/{:02d}, Train, Loss {:9.5f}, Mesh {:9.5f}, Param {:9.5f}, CMap {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f}, Diff {:9.5f} ,Best Val-loss: {:9.5f}".format(
        epoch, args.epochs,
        sum(logs['loss']) / len(logs['loss']),
        sum(logs['recon_loss']) / len(logs['recon_loss']),
        sum(logs['param_loss']) / len(logs['param_loss']),
        sum(logs['cmap_loss']) / len(logs['cmap_loss']),
        sum(logs['cmap_consistency']) / len(logs['cmap_consistency']),
        sum(logs['penetr_loss']) / len(logs['penetr_loss']),
        sum(logs['diff_loss']) / len(logs['diff_loss']),
        best_train_loss
    )
    logger(out_str)

    if mean_recon_loss <= best_train_loss:
        save_name = os.path.join(
            checkpoint_root, 'model_best_{}_diffusion.pth'.format(mode))
        torch.save({
            'network': model.state_dict(),
            'epoch': epoch
        }, save_name)

        save_name = os.path.join(
            checkpoint_root, 'model_best_{}_ae.pth'.format(mode))
        torch.save({
            'network': autoencoder.state_dict(),
            'epoch': epoch
        }, save_name)

        save_name = os.path.join(
            checkpoint_root, 'model_best_{}_adapt_layer.pth'.format(mode))
        torch.save({
            'network': adapt_layer.state_dict(),
            'epoch': epoch
        }, save_name)


    return best_train_loss


def val(args, writer, epoch, autoencoder, model, adapt_layer, train_loader, device, optimizer, logger, checkpoint_root, best_val_loss, rh_mano, rh_faces, mode='val'):
   # validation
    logs = defaultdict(list)
    autoencoder.eval()
    model.eval()
    adapt_layer.eval()
    a, b, c, d, e = args.weight
    with torch.no_grad():
        # obj_language  is   str
        for batch_idx, input in enumerate(train_loader):
            obj_pc = input["obj_pc"].to(device)
            hand_param = input["hand_param"].to(device)
            if args.dataset == "oakink":
                gt_mano = rh_mano(hand_param[:, 10:58], hand_param[:, :10])
                hand_xyz = gt_mano.verts.to(
                    device) + hand_param[:, None, 58:]  # [B,778,3]
            else:
                th_v_template = input["th_v_template"].to(device)
                hand_xyz = input["hand_verts"].to(device)

            hand_feature, _, _ = autoencoder.hand_encoder(
                hand_xyz.permute(0, 2, 1))
            z = autoencoder.encoder(hand_feature).view(
                hand_feature.shape[0], -1, 3)  # [B , 256 , 3] <- [B, 768]

            diff_loss, predict_x = model(z, obj_feature=obj_pc, language_feature=None)


            adapt_x = adapt_layer(predict_x.reshape(
                predict_x.shape[0], -1).to(device), obj_pc)

            recon_param = autoencoder.decoder(predict_x.reshape(
                predict_x.shape[0], -1).to(device) + adapt_x)

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

            obj_nn_dist_gt, obj_nn_idx_gt = utils_loss.get_NN(
                obj_pc.permute(0, 2, 1)[:, :, :3], hand_xyz)
            obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(
                obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz)


            param_loss = torch.nn.functional.mse_loss(recon_param, hand_param, reduction='none').sum() / recon_param.size(0)


            recon_loss_num, _ = chamfer_distance(
                recon_xyz, hand_xyz, point_reduction='sum', batch_reduction='mean')
            cmap_loss = CMap_loss3(obj_pc.permute(
                0, 2, 1)[:, :, :3], recon_xyz, obj_nn_dist_recon < 0.01**2)
            consistency_loss = CMap_consistency_loss(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz, hand_xyz,
                                                     obj_nn_dist_recon, obj_nn_dist_gt)
            # inter penetration loss
            rh_face = rh_faces[0]
            rh_faces = rh_face.expand(
                recon_xyz.shape[0], -1, -1)  # align the B
            penetr_loss = inter_penetr_loss(recon_xyz, rh_faces, obj_pc.permute(0, 2, 1)[
                                            :, :, :3], obj_nn_dist_recon, obj_nn_idx_recon)

            loss = a * recon_loss_num + b * param_loss + c * \
                cmap_loss + d * penetr_loss + e * consistency_loss
            logs['loss'].append(loss.item())
            logs['diff_loss'].append(diff_loss)
            logs['recon_loss'].append(recon_loss_num)
            logs['param_loss'].append(param_loss.item())
            logs['cmap_loss'].append(cmap_loss.item())
            logs['penetr_loss'].append(penetr_loss.item())
            logs['cmap_consistency'].append(consistency_loss.item())

    writer.add_scalar('loss', sum(logs['loss']) / len(logs['loss']), epoch)
    writer.add_scalar('diff_loss', sum(
        logs['diff_loss']) / len(logs['diff_loss']), epoch)
    writer.add_scalar('recon_loss', sum(
        logs['recon_loss']) / len(logs['recon_loss']), epoch)
    writer.add_scalar('param_loss', sum(
        logs['param_loss']) / len(logs['param_loss']), epoch)
    writer.add_scalar('cmap_loss', sum(
        logs['cmap_loss']) / len(logs['cmap_loss']), epoch)
    writer.add_scalar('penetr_loss', sum(
        logs['penetr_loss']) / len(logs['penetr_loss']), epoch)
    writer.add_scalar('cmap_consistency', sum(
        logs['cmap_consistency']) / len(logs['cmap_consistency']), epoch)


    val_loss = sum(logs['penetr_loss']) / len(logs['penetr_loss'])

    best_val_loss = min(val_loss , best_val_loss)
        

    out_str = "Epoch: {:02d}/{:02d}, {}  , loss {:9.5f}, Mesh {:9.5f}, Param {:9.5f}, CMap {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f}, Diff {:9.5f} ,Best Val-loss: {:9.5f}".format(
        epoch, args.epochs,mode,
        sum(logs['loss']) / len(logs['loss']),
        sum(logs['recon_loss']) / len(logs['recon_loss']),
        sum(logs['param_loss']) / len(logs['param_loss']),
        sum(logs['cmap_loss']) / len(logs['cmap_loss']),
        sum(logs['cmap_consistency']) / len(logs['cmap_consistency']),
        sum(logs['penetr_loss']) / len(logs['penetr_loss']),
        sum(logs['diff_loss']) / len(logs['diff_loss']),
        best_val_loss
    )
    logger(out_str)

    if val_loss <= best_val_loss:
        save_name = os.path.join(
            checkpoint_root, 'model_best_{}_diffusion.pth'.format(mode))
        torch.save({
            'network': model.state_dict(),
            'epoch': epoch
        }, save_name)

        save_name = os.path.join(
            checkpoint_root, 'model_best_{}_ae.pth'.format(mode))
        torch.save({
            'network': autoencoder.state_dict(),
            'epoch': epoch
        }, save_name)

        save_name = os.path.join(
            checkpoint_root, 'model_best_{}_adapt_layer.pth'.format(mode))
        torch.save({
            'network': adapt_layer.state_dict(),
            'epoch': epoch
        }, save_name)

    return best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config, 'r') as configfile:
        config = json.load(configfile)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()

    save_root = os.path.join('./logs', f"adapt/{args.dataset}/{args.file_name}")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    shutil.copy(args.config, save_root)

    log_root = save_root + '/exp.log'
        # logger
    logger = makelogger(makepath(os.path.join(log_root), isfile=True)).info
    starttime = datetime.now().replace(microsecond=0)
    logger('Started training %s' % (starttime))
    gpu_brand = torch.cuda.get_device_name(0) if args.use_cuda else None
    if args.use_cuda and torch.cuda.is_available():
        logger('Using 1 CUDA cores [%s] for training!' % (gpu_brand))
    logger(args)
    logger(args.weight)

    set_random_seed(args.seed)

    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_num = 1
    """load autoencoder"""
    autoencoder = Autoencoder(
        args=args,
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_decoder_sizes=args.decoder_layer_sizes)
    checkpoint = torch.load(args.autoencoder_path,
                            map_location=torch.device(device))
    new_state_dict = {}
    for key, value in checkpoint['network'].items():
        if key.startswith('module.'):
            new_key = key[7:]  
        else:
            new_key = key
        new_state_dict[new_key] = value
    autoencoder.load_state_dict(new_state_dict)
    autoencoder = autoencoder.to(device)
    """load diffusion"""

    with open(args.diffusion_config, 'r') as f:
        diffusion_param = json.load(f)
    net = PointNet2SemSegSSG(diffusion_param)


    diffusion_model = DDIM(args, nn_model=net, betas=(
        1e-4, 0.02), n_T=1000, device=device, drop_prob=0.1)

    checkpoint = torch.load(args.diffusion_path,
                            map_location=torch.device(device))
    new_state_dict = {}
    for key, value in checkpoint['network'].items():
        if key.startswith('module.'):
            new_key = key[7:]  
        else:
            new_key = key
        new_state_dict[new_key] = value
    diffusion_model.load_state_dict(new_state_dict)
    diffusion_model = diffusion_model.to(device)
    """init AdaptLayer"""
    # adapt_layer = AdaptLayer(args.batch_size).to(device)
    adapt_layer = AdaptLayer2(args.adapt_layer["layer_sizes"], args.adapt_layer["latent_size"],conditional=args.adapt_layer["conditional"], condition_size=args.adapt_layer["condition_size"] , intention = args.adapt_layer["intention"]).to(device)
    # adapt_layer = AdaptLayer3(args.adapt_layer["layer_sizes"], args.adapt_layer["latent_size"] , conditional= args.adapt_layer["conditional"] ,condition_size= args.adapt_layer["condition_size"]).to(device)
    # adapt_layer =AdaptLayerTransformer(args.adapt_layer["layer_sizes"], args.adapt_layer["latent_size"],conditional=args.adapt_layer["conditional"], condition_size=args.adapt_layer["condition_size"]).to(device)
    logger(adapt_layer)

    if 'Train' in args.train_mode:
        if args.dataset == "oakink":
            train_dataset = Oakink(mode="train", batch_size=args.batch_size, args=args)
        else:
            train_dataset = Grab(mode="train", batch_size=args.batch_size, args=args,aug_ratio = args.aug_ratio)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)

    if 'Val' in args.train_mode:
        if args.dataset == "oakink":
            val_dataset = Oakink(mode="val", batch_size=args.batch_size, args=args)
        else:
            val_dataset = Grab(mode="val", batch_size=args.batch_size, args=args,aug_ratio = args.aug_ratio)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.dataloader_workers)



    for param in autoencoder.autoencoder.parameters():
        param.requires_grad = False
    for param in diffusion_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(adapt_layer.parameters(), lr=args.learning_rate)
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [10,50,100]], gamma=0.5)
    train_writer = SummaryWriter(os.path.join(
        os.path.dirname(log_root), 'tensorboard/train'))
    val_writer = SummaryWriter(os.path.join(
        os.path.dirname(log_root), 'tensorboard/val'))
    test_writer = SummaryWriter(os.path.join(
        os.path.dirname(log_root), 'tensorboard/test'))
    # mano hand model
    with torch.no_grad():
        rh_mano = ManoLayer(center_idx=0, mano_assets_root="/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE/assets/mano_v1_2").to(device)
    # [1, 1538, 3], face triangle indexes
    rh_faces = rh_mano.th_faces.view(1, -1, 3).contiguous()
    rh_faces = rh_faces.repeat(args.batch_size, 1, 1).to(device)  # [N, 1538, 3]
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_eval_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        if 'Train' in args.train_mode:
            best_train_loss = train(args, train_writer, epoch, autoencoder, diffusion_model, adapt_layer, train_loader,
                                    device, optimizer, logger, save_root, best_train_loss, rh_mano, rh_faces , "train")
            scheduler.step()
        if 'Val' in args.train_mode:
            best_val_loss = val(args, val_writer, epoch, autoencoder, diffusion_model, adapt_layer, val_loader,
                                device, optimizer, logger, save_root, best_val_loss, rh_mano, rh_faces, "val")

