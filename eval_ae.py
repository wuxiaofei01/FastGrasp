import os
import time
import torch
import argparse
import json
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset.dataset_eval import Oakink ,Grab
from network.autoencoder.autoencoder import Autoencoder
import numpy as np
import random
from utils import utils_loss
from utils.utils import convert_euler_to_rotmat
from utils.loss import CVAE_loss_mano, CMap_loss, CMap_loss1, CMap_loss3, CMap_loss4, inter_penetr_loss, CMap_consistency_loss, set_random_seed
from pytorch3d.loss import chamfer_distance
import mano
import statistics
# from manotorch.manolayer import ManoLayer, MANOOutput
from evaluation.displacement import grasp_displacement ,diversity
from evaluation.vis import vis_dataset
from tqdm import tqdm
import ipdb
from manotorch.manolayer import ManoLayer, MANOOutput
from manopth.manolayer import grabManoLayer
grab_mano_layer = grabManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("assets/mano_v1_2/models"), use_pca=False, joint_rot_mode="rotmat").to("cuda")
# grab_mano_layer = grabManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("assets/mano_v1_2/models"), use_pca=False, joint_rot_mode="axisang").to("cuda")

def val(args, model, val_loader, device, rh_mano, rh_faces, mode='val'):
    # validation
    cluster = []
    model.eval()
    simulation_displacements_list = []  # std
    penetration_distances_list = []
    intersection_volumes_list = []
    '''accelerate evaluation '''
    obj_face_list = []
    obj_vert_list = []
    hand_out_list = []
    hand_face_list = []

    model_path = f"{os.path.dirname(args.model_path)}"
    ply_path = model_path +f"/ply_{args.dataset}"
    if not os.path.exists(ply_path):
        os.makedirs(ply_path)

    eval_path = model_path +f"/eval_{args.dataset}/"
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    with torch.no_grad():
        for batch_idx, input in enumerate(tqdm(val_loader)):
            obj_pc = input["obj_pc"].to(device)
            hand_param = input["hand_param"].to(device)
            obj_face = input["obj_face"]
            obj_vert = input["obj_vert"]
            if args.dataset =="oakink":
                gt_mano = rh_mano(hand_param[:, 10:58] , hand_param[:,:10])
                gt_hand = gt_mano.verts.to(device) +  hand_param[:,None, 58:] # [B,778,3]
            else:
                th_v_template = input["th_v_template"].to(device)
                gt_hand = input["hand_verts"].to(device)
                hand_faces = input["hand_faces"].squeeze(0)

            '''autoencoder'''
            recon_param = model(gt_hand.permute(0,2,1))  # recon [B,61] mano params

            if args.mano =="grab":
                handbetas = hand_param[:,:10]
                handeuler = recon_param[:, 10:58].view(-1,16,3)  # [ B , 16 , 3 ]
                handrot = convert_euler_to_rotmat(handeuler)# [ B , 16 , 3 , 3]
                th_trans =recon_param[:, 58:] # [ B , 3]
                hand_verts, _ ,_= grab_mano_layer(handrot[0].unsqueeze(0).to(device),th_betas=handbetas[0].unsqueeze(0).to(device), th_trans=th_trans[0].unsqueeze(0).to(device), th_v_template=th_v_template[0].unsqueeze(0))
                hand_faces = grab_mano_layer.th_faces

                if batch_idx <= 300 or (batch_idx % 477 == 0):      
                    vis_dataset(obj_face , obj_vert, hand_verts.to("cpu"), hand_faces.to("cpu") , f"{ply_path}/{batch_idx}.ply")
            elif args.mano =="oakink":
                recon_mano = rh_mano(recon_param[:, 10:58] , recon_param[:,:10])
                hand_verts = recon_mano.verts.to(device) +  recon_param[:,None, 58:] # [B,778,3]
                hand_faces = rh_mano.th_faces

                if batch_idx <= 300 or (batch_idx % 477 == 0): 
                    vis_dataset(obj_face , obj_vert, hand_verts.to("cpu"), hand_faces.to("cpu") , f"{ply_path}/{batch_idx}.ply")
            cluster.append(recon_param.squeeze(0).cpu().numpy())
            ''''simulation_displacement , penetration_distance , intersection_volume '''
            if batch_idx  % 256  == 0 and batch_idx != 0:
                simulation_displacement , penetration_distance , intersection_volume = grasp_displacement(obj_face_list , obj_vert_list,hand_out_list,  hand_face_list , "")
                simulation_displacements_list += simulation_displacement
                penetration_distances_list += penetration_distance
                intersection_volumes_list += intersection_volume 

                obj_face_list.clear()
                obj_vert_list.clear()
                hand_out_list.clear()
                hand_face_list.clear()
            obj_face_list.append(obj_face.squeeze(0))
            obj_vert_list.append(obj_vert.squeeze(0))
            hand_out_list.append(hand_verts.squeeze(0))
            hand_face_list.append(hand_faces)
    cluster_array = np.array(cluster)
    entropy, cluster_size = diversity(cluster_array, cls_num=20)
    # np.save("baseline_train_train_hand.npy", cluster_array)
    '''maybe datalen % 16 != 0 ,so deal it'''
    simulation_displacement , penetration_distance , intersection_volume = grasp_displacement(obj_face_list , obj_vert_list,hand_out_list,  hand_face_list , "")

    simulation_displacements_list += simulation_displacement
    penetration_distances_list += penetration_distance
    intersection_volumes_list += intersection_volume 

    std_simulation_displacement = statistics.stdev(simulation_displacements_list)
    mean_simulation_displacement = sum(simulation_displacements_list) / len(simulation_displacements_list)
    mean_penetration_distance = sum(penetration_distances_list) / len(penetration_distances_list)
    mean_intersection_volume = sum(intersection_volumes_list) / len(intersection_volumes_list)
    contact_ratio= np.mean(np.array(intersection_volumes_list) != 0)


    with open(eval_path+f"val.txt", "w", encoding="utf-8") as file:
        print(f"mean_simulation_displacement : {mean_simulation_displacement * 1e2:.4f}e-02\n"
        f"std_simulation_displacement : {std_simulation_displacement * 1e2:.4f}e-02\n"
        f"mean_penetration_distance : {mean_penetration_distance * 1e2:.4f}e-02\n"
        f"mean_intersection_volume : {mean_intersection_volume * 1e6:.4f}e-06\n"
        f"contact_ratio : {contact_ratio * 1e2 :.4f}e-02\n"
        f" entropy :, {entropy} \n"
        f"cluster_size : {cluster_size}",file=file)

    print(f"mean_simulation_displacement : {mean_simulation_displacement * 1e2:.4f}e-02\n"
    f"std_simulation_displacement : {std_simulation_displacement * 1e2:.4f}e-02\n"
    f"mean_penetration_distance : {mean_penetration_distance * 1e2:.4f}e-02\n"
    f"mean_intersection_volume : {mean_intersection_volume * 1e6:.4f}e-06\n"
    f"contact_ratio : {contact_ratio * 1e2 :.4f}e-02\n"
    f" entropy :, {entropy} \n"
    f"cluster_size : {cluster_size}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config, 'r') as configfile:
        config = json.load(configfile)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()

    
    # seed
    set_random_seed(args.seed)
    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device", device)
    device_num = 1

    model = Autoencoder(
        args = args,
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_decoder_sizes=args.decoder_layer_sizes).to(device)


    # network
    checkpoint = torch.load(args.model_path, map_location=torch.device(device))
    new_state_dict = {}
    for key, value in checkpoint['network'].items():
        if key.startswith('module.'):
            new_key = key[7:]  # 去除前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    # multi-gpu
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model)
            device_num = len(device_ids)

    if 'Test' in args.train_mode:
        if args.dataset =="oakink":
            eval_dataset = Oakink(mode="test", batch_size=args.batch_size , args = args)
        else:
            eval_dataset = Grab(mode="test", batch_size=args.batch_size , args = args)
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)
    # mano hand model
    print(args.model_path)
    with torch.no_grad():
        rh_mano = ManoLayer(center_idx=0, mano_assets_root="assets/mano_v1_2").to(device)
    rh_faces = rh_mano.th_faces.view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes
    rh_faces = rh_faces.repeat(args.batch_size, 1, 1).to(device) # [N, 1538, 3]
    if 'Test' in args.train_mode:
        val(args, model, eval_loader, device, rh_mano, rh_faces, 'test')
    else:
        print("no dataset!")

