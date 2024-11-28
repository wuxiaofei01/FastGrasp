import os
import time
import torch
import argparse
import json
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset.oakink_dataset import Oakink
from dataset.grab_dataset import Grab
from network.autoencoder.autoencoder import Autoencoder
import numpy as np
import random
from utils import utils_loss
from utils.utils import  makepath, makelogger , convert_euler_to_rotmat
from utils.loss import CVAE_loss_mano, CMap_loss, CMap_loss1, CMap_loss3, CMap_loss4, inter_penetr_loss, CMap_consistency_loss, set_random_seed,kl_div_normal
from pytorch3d.loss import chamfer_distance
import mano
import ipdb
import sys
from torch.utils.tensorboard import SummaryWriter
from evaluation.vis import vis_hand
from manopth.manolayer import grabManoLayer
from manotorch.manolayer import ManoLayer, MANOOutput
from datetime import datetime

grab_mano_layer = grabManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("assets/mano_v1_2/models"), use_pca=False, joint_rot_mode="rotmat").to("cuda")


def train(args, writer , epoch, model, train_loader, device, optimizer, logger, checkpoint_root ,best_train_loss, rh_mano, rh_faces):
    since = time.time()
    logs = defaultdict(list)
    a, b, c, d, e , f= args.weight
    model.train()
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
        hand_feature, _, _ = model.hand_encoder(hand_xyz.permute(0,2,1))
        # pointnet
        z = model.encoder(hand_feature)

        recon_param = model.decoder(z)

        if args.dataset =="oakink":
            recon_mano = rh_mano(recon_param[:, 10:58] , recon_param[:,:10])
            recon_xyz = recon_mano.verts.to(device) +  recon_param[:,None, 58:] # [B,778,3]
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
        obj_nn_dist_gt, obj_nn_idx_gt = utils_loss.get_NN(obj_pc.permute(0,2,1)[:,:,:3], hand_xyz)
        obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz)
        # mano param loss
        param_loss = torch.nn.functional.mse_loss(recon_param, hand_param, reduction='none').sum() / recon_param.size(0)
        # mano recon xyz loss, KLD loss
        recon_loss_num, _ = chamfer_distance(recon_xyz, hand_xyz, point_reduction='sum', batch_reduction='mean')

        #cmap_loss = CMap_loss(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, obj_cmap)
        cmap_loss = CMap_loss3(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, obj_nn_dist_recon < 0.01**2)
        # cmap consistency loss
        consistency_loss = CMap_consistency_loss(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, hand_xyz,
                                                 obj_nn_dist_recon, obj_nn_dist_gt)
        # inter penetration loss
        rh_face = rh_faces[0] 
        rh_faces = rh_face.expand(recon_xyz.shape[0], -1, -1) # align the B
        penetr_loss =inter_penetr_loss(recon_xyz, rh_faces, obj_pc.permute(0,2,1)[:,:,:3],
                                        obj_nn_dist_recon, obj_nn_idx_recon)

        kl_loss = kl_div_normal(z)


        if epoch >= 5:
            loss = a * recon_loss_num + b * param_loss + c * cmap_loss + d * penetr_loss + e * consistency_loss + f * kl_loss
        else:
            loss = a * recon_loss_num + b * param_loss + d * penetr_loss + e * consistency_loss + f * kl_loss
        loss.backward()
        optimizer.step()
        
        logs['recon_loss'].append(recon_loss_num)
        logs['loss'].append(loss.item())
        logs['param_loss'].append(param_loss.item())
        logs['cmap_loss'].append(cmap_loss.item())
        logs['penetr_loss'].append(penetr_loss.item())
        logs['cmap_consistency'].append(consistency_loss.item())
        logs['kl_loss'].append(kl_loss.item())

    writer.add_scalar('loss', sum(logs['loss']) / len(logs['loss']) , epoch)
    writer.add_scalar('recon_loss', sum(logs['recon_loss']) / len(logs['recon_loss']) , epoch)
    writer.add_scalar('param_loss', sum(logs['param_loss']) / len(logs['param_loss']) , epoch)
    writer.add_scalar('cmap_loss', sum(logs['cmap_loss']) / len(logs['cmap_loss']) , epoch)
    writer.add_scalar('penetr_loss', sum(logs['penetr_loss']) / len(logs['penetr_loss']), epoch)
    writer.add_scalar('cmap_consistency', sum(logs['cmap_consistency']) / len(logs['cmap_consistency']) , epoch)
    writer.add_scalar('kl_loss', sum(logs['kl_loss']) / len(logs['kl_loss']) , epoch)

    mean_recon_loss = sum(logs['param_loss']) / len(logs['param_loss']) 
        
    out_str = "Epoch: {:02d}/{:02d}, train, Mean Toal Loss {:9.5f}, Mesh {:9.5f}, Param {:9.5f}, CMap {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f} , KL {:9.5f} , Best param-loss: {:9.5f}".format(
        epoch, args.epochs,
        sum(logs['loss']) / len(logs['loss']),
        sum(logs['recon_loss']) / len(logs['recon_loss']),
        sum(logs['param_loss']) / len(logs['param_loss']),
        sum(logs['cmap_loss']) / len(logs['cmap_loss']),
        sum(logs['cmap_consistency']) / len(logs['cmap_consistency']),
        sum(logs['penetr_loss']) / len(logs['penetr_loss']),
        sum(logs['kl_loss']) / len(logs['kl_loss']),
        min(best_train_loss, mean_recon_loss)
    )
    logger(out_str)
    if mean_recon_loss < best_train_loss :
        save_name = os.path.join(checkpoint_root, 'model_best_train.pth')
        torch.save({
            'network': model.state_dict(),
            'epoch': epoch
        }, save_name)

    return min(mean_recon_loss , best_train_loss)

def val(args,writer, epoch, model, val_loader, device,logger, checkpoint_root, best_val_loss, rh_mano, rh_faces, mode='val'):
    # validation
    model.eval()
    a, b, c, d, e ,f  = args.weight
    logs = defaultdict(list)
    with torch.no_grad():
        for batch_idx, input in enumerate(val_loader):
            obj_pc = input["obj_pc"].to(device)
            hand_param = input["hand_param"].to(device)
            if args.dataset =="oakink":
                gt_mano = rh_mano(hand_param[:, 10:58] , hand_param[:,:10])
                hand_xyz = gt_mano.verts.to(device) +  hand_param[:,None, 58:] # [B,778,3]
            else:
                th_v_template = input["th_v_template"].to(device)
                hand_xyz = input["hand_verts"].to(device)

            hand_feature, _, _ = model.hand_encoder(hand_xyz.permute(0,2,1))
            # pointnet
            z = model.encoder(hand_feature)

            recon_param = model.decoder(z)

            if args.dataset == 'oakink':
                recon_mano = rh_mano(recon_param[:, 10:58] , recon_param[:,:10])
                recon_xyz = recon_mano.verts.to(device) +  recon_param[:,None, 58:] # [B,778,3]
            else:
                handbetas = hand_param[:,:10]
                handeuler = recon_param[:, 10:58].view(-1,16,3)  # [ B , 16 , 3 ]
                handrot = convert_euler_to_rotmat(handeuler)# [ B , 16 , 3 , 3]
                th_trans =recon_param[:, 58:] # [ B , 3]
                hand_verts_list = []
                for k , x in enumerate(handrot):
                    hand_vert, _ ,_= grab_mano_layer(handrot[k].unsqueeze(0).to(device),th_betas=handbetas[k].unsqueeze(0).to(device), th_trans=th_trans[k].unsqueeze(0).to(device), th_v_template=th_v_template[k].unsqueeze(0))
                    hand_verts_list.append(hand_vert)
                recon_xyz = torch.stack(hand_verts_list).squeeze(1)

            obj_nn_dist_gt, obj_nn_idx_gt = utils_loss.get_NN(obj_pc.permute(0,2,1)[:,:,:3], hand_xyz)
            obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz)

            # mano param loss
            param_loss = torch.nn.functional.mse_loss(recon_param, hand_param, reduction='none').sum() / recon_param.size(0)
            recon_loss_num, _ = chamfer_distance(recon_xyz, hand_xyz, point_reduction='sum', batch_reduction='mean')
            cmap_loss = CMap_loss3(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, obj_nn_dist_recon < 0.01**2)
            consistency_loss = CMap_consistency_loss(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, hand_xyz,
                                                    obj_nn_dist_recon, obj_nn_dist_gt)
            # inter penetration loss
            rh_face = rh_faces[0] 
            rh_faces = rh_face.expand(recon_xyz.shape[0], -1, -1) # align the B
            penetr_loss =inter_penetr_loss(recon_xyz, rh_faces, obj_pc.permute(0,2,1)[:,:,:3],obj_nn_dist_recon, obj_nn_idx_recon)
            kl_loss = kl_div_normal(z)


            if epoch >= 5:
                loss = a * recon_loss_num + b * param_loss + c * cmap_loss + d * penetr_loss + e * consistency_loss + f * kl_loss
            else:
                loss = a * recon_loss_num + b * param_loss + d * penetr_loss + e * consistency_loss  + f * kl_loss
            
            logs['recon_loss'].append(recon_loss_num)
            logs['loss'].append(loss.item())
            logs['param_loss'].append(param_loss.item())
            logs['cmap_loss'].append(cmap_loss.item())
            logs['penetr_loss'].append(penetr_loss.item())
            logs['cmap_consistency'].append(consistency_loss.item())
            logs['kl_loss'].append(kl_loss.item())

        writer.add_scalar('loss', sum(logs['loss']) / len(logs['loss']) , epoch)
        writer.add_scalar('recon_loss', sum(logs['recon_loss']) / len(logs['recon_loss']) , epoch)
        writer.add_scalar('param_loss', sum(logs['param_loss']) / len(logs['param_loss']) , epoch)
        writer.add_scalar('cmap_loss', sum(logs['cmap_loss']) / len(logs['cmap_loss']) , epoch)
        writer.add_scalar('penetr_loss', sum(logs['penetr_loss']) / len(logs['penetr_loss']), epoch)
        writer.add_scalar('cmap_consistency', sum(logs['cmap_consistency']) / len(logs['cmap_consistency']) , epoch)
        writer.add_scalar('kl_loss', sum(logs['kl_loss']) / len(logs['kl_loss']) , epoch)

    val_loss = sum(logs['param_loss']) / len(logs['param_loss'])
    out_str = "Epoch: {:02d}/{:02d},  {} , Mean Toal Loss {:9.5f}, Mesh {:9.5f}, Param {:9.5f}, CMap {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f} , KL {:9.5f} , Best param-loss: {:9.5f}".format(
        epoch, args.epochs,mode,
        sum(logs['loss']) / len(logs['loss']),
        sum(logs['recon_loss']) / len(logs['recon_loss']),
        sum(logs['param_loss']) / len(logs['param_loss']),
        sum(logs['cmap_loss']) / len(logs['cmap_loss']),
        sum(logs['cmap_consistency']) / len(logs['cmap_consistency']),
        sum(logs['penetr_loss']) / len(logs['penetr_loss']),
        sum(logs['kl_loss']) / len(logs['kl_loss']),
        min(best_val_loss, val_loss)
    )
    logger(out_str)
    if val_loss < best_val_loss:
        save_name = os.path.join(checkpoint_root, 'model_best_{}.pth'.format(mode))
        torch.save({
            'network': model.state_dict(),
            'epoch': epoch
        }, save_name)

    return min(best_val_loss, val_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config, 'r') as configfile:
        config = json.load(configfile)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()

     # log file
    save_root = os.path.join('./logs', f"autoencoder/{args.dataset}/{args.file_name}" )
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
    logger(args.weight)

    # seed
    set_random_seed(args.seed)
    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_num = 1
    model = Autoencoder(
        args = args,
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_decoder_sizes=args.decoder_layer_sizes).to(device)

    # multi-gpu
    if device == torch.device("cuda"):
        device_ids = range(torch.cuda.device_count())
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model)
            device_num = len(device_ids)
    # dataset

    if 'Train' in args.train_mode:
        if args.dataset =="oakink":
            train_dataset = Oakink(mode="train", batch_size=args.batch_size , args = args)
        else:
            train_dataset = Grab(mode="train", batch_size=args.batch_size , args = args , aug_ratio = args.aug_ratio)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers,drop_last=True )
    if 'Val' in args.train_mode:
        if args.dataset =="oakink":
            val_dataset = Oakink(mode="val", batch_size=args.batch_size  , args = args)
        else:
            val_dataset = Grab(mode="val", batch_size=args.batch_size  , args = args)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)
    if 'Test' in args.train_mode:
        if args.dataset =="oakink":
            eval_dataset = Oakink(mode="test", batch_size=args.batch_size , args = args)
        else:
            eval_dataset = Grab(mode="test", batch_size=args.batch_size , args = args)
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.1, 0.2, 0.3, 0.5]], gamma=0.5)
    train_writer = SummaryWriter(os.path.join(
        os.path.dirname(log_root), 'tensorboard/train'))
    val_writer = SummaryWriter(os.path.join(
        os.path.dirname(log_root), 'tensorboard/val'))
    test_writer = SummaryWriter(os.path.join(
        os.path.dirname(log_root), 'tensorboard/test'))

    # mano hand model
    with torch.no_grad():
        rh_mano = ManoLayer(
            center_idx=0, mano_assets_root="assets/mano_v1_2").to(device)
    # [1, 1538, 3], face triangle indexes
    rh_faces = rh_mano.th_faces.view(1, -1, 3).contiguous()
    rh_faces = rh_faces.repeat(args.batch_size, 1, 1).to(device)  # [N, 1538, 3]
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_eval_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        if 'Train' in args.train_mode:
            best_train_loss = train(args, train_writer, epoch, model, train_loader,
                                    device, optimizer, logger, save_root, best_train_loss, rh_mano, rh_faces,)
            scheduler.step()

        if 'Val' in args.train_mode:
            best_val_loss = val(args, val_writer, epoch, model, val_loader,
                                device, logger, save_root, best_val_loss, rh_mano, rh_faces, 'val')

        if 'Test' in args.train_mode:
            best_eval_loss = val(args, test_writer, epoch, model, eval_loader,
                                device, logger, save_root, best_eval_loss, rh_mano, rh_faces, 'test')
