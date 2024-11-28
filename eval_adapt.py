import os
import time
import torch
import argparse
import json
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset.dataset_eval import Oakink, Grab
from network.autoencoder.autoencoder import Autoencoder
from network.adapt_layer.adapt_layer import AdaptLayer , AdaptLayer2 , AdaptLayer3
from network.diffusion.ddim import DDIM
from network.diffusion.pointnet2.pointnet2_ssg_sem import PointNet2SemSegSSG
from utils.utils import convert_euler_to_rotmat
import numpy as np
import random
from utils import utils_loss
from utils.loss import CVAE_loss_mano, CMap_loss, CMap_loss1, CMap_loss3, CMap_loss4, inter_penetr_loss, CMap_consistency_loss, set_random_seed
from pytorch3d.loss import chamfer_distance
import mano
import statistics
from evaluation.converter import transform_to_canonical, convert_joints
from evaluation.displacement import grasp_displacement,diversity
from evaluation.halo import intersection_eval
from evaluation.vis import vis_dataset
from tqdm import tqdm
import ipdb
from manotorch.manolayer import ManoLayer, MANOOutput
from manopth.manolayer import grabManoLayer
import shutil
import trimesh
from torch.utils.data.sampler import SubsetRandomSampler
grab_mano_layer = grabManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("assets/mano_v1_2/models"), use_pca=False, joint_rot_mode="rotmat").to("cuda")

def sample_type(args , diffusion_model , guide_w , tensor_pc , device):

    if args.intention == False:
        if args.sample == "ddpm":
            z_1 = diffusion_model.sample(tensor_pc.shape[0], (256, 3), device, guide_w= args.guide_w,obj_feature=tensor_pc ).view(tensor_pc.shape[0], -1)
        else:
            z_1 = diffusion_model.sample_ddim(tensor_pc.shape[0], (256, 3), device, guide_w= args.guide_w ,obj_feature=tensor_pc, ddim_step= args.ddim_step ).view(tensor_pc.shape[0], -1)
    else:
        z_1 = diffusion_model.sample_ddim(tensor_pc.shape[0], (256, 3), device, guide_w=2,obj_feature=tensor_pc ,language_feature=tensor_language ).view(tensor_pc.shape[0], -1)
    return z_1

def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign
def val(args, autoencoder, diffusion_model,adapt_layer ,val_loader, device, rh_mano, rh_faces, mode='val'):
    # validation
    cluster = []
    cluster2 = []
    diffusion_model.eval()
    autoencoder.eval()
    adapt_layer.eval()
    simulation_displacements_list = []  # 获取 std
    penetration_distances_list = []
    intersection_volumes_list = []
    contact_list = []
    '''加速 displacement ,多个batch进行加速'''
    pc_list = []
    obj_face_list = []
    obj_vert_list = []
    hand_out_list = []
    hand_face_list = []
    gt_hand_verts = []
    gt_hand_param = []
    language_list = []
    th_v_template_list = []

    obj_mesh_path_list = []
    model_path = f"{os.path.dirname(args.diffusion_path)}"
    ply_path = model_path + f"/ply_{args.guide_w}"
    if not os.path.exists(ply_path):
        os.makedirs(ply_path)

    eval_path = model_path + "/eval/"
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    
    '''移动checkpoint到val目录下'''
    shutil.copy(args.diffusion_path, eval_path)
    shutil.copy(args.autoencoder_path, eval_path)
    shutil.copy(args.adapt_path, eval_path)

    flag = 0
    for batch_idx, input in enumerate(tqdm(val_loader)):
        obj_pc = input["obj_pc"].to(device)
        hand_param = input["hand_param"].to(device)
        obj_face = input["obj_face"]
        obj_vert = input["obj_vert"]
        if args.dataset =="oakink":
            gt_mano = rh_mano(hand_param[:, 10:58] , hand_param[:,:10])
            gt_hand = gt_mano.verts.to(device) +  hand_param[:,None, 58:] # [B,778,3]
            hand_faces = rh_mano.th_faces
        else:
            th_v_template = input["th_v_template"].to(device)
            gt_hand = input["hand_verts"].to(device)
            hand_faces = input["hand_faces"].squeeze(0)
        '''gt'''
        pc_list.append(obj_pc)
        obj_face_list.append(obj_face.squeeze(0))
        obj_vert_list.append(obj_vert.squeeze(0))
        hand_face_list.append(hand_faces)
        gt_hand_verts.append(gt_hand)
        gt_hand_param.append(hand_param)
        
        # vis_dataset(obj_face, obj_vert, gt_hand.to("cpu"), hand_faces.to("cpu"), f"{ply_path}/gt_{batch_idx}.ply")
        # if batch_idx >100:
        #     exit()
    
        if batch_idx + 1 == len(val_loader) or len(pc_list)==32:

            with torch.no_grad():
                tensor_pc = torch.cat(pc_list, dim=0)
                if args.dataset == "grab":
                    sample_times = 20
                    tensor_pc = tensor_pc.repeat_interleave(sample_times, dim=0)
                    th_v_template_list = [item for item in th_v_template_list for _ in range(sample_times)]
                    obj_vert_list = [item for item in obj_vert_list for _ in range(sample_times)]
                    obj_face_list = [item for item in obj_face_list for _ in range(sample_times)]
                    hand_face_list = [item for item in hand_face_list for _ in range(sample_times)]

                z_1 = sample_type(args , diffusion_model , args.guide_w , tensor_pc , device)

                # z_1 = diffusion_model.sample(tensor_pc.shape[0], (256, 3), device, guide_w=args.guide_w,obj_feature=tensor_pc ).view(tensor_pc.shape[0], -1)
                # z_1 = diffusion_model.sample_ddim(tensor_pc.shape[0], (256, 3), device, guide_w=args.guide_w,obj_feature=tensor_pc,ddim_step= args.ddim_step ).view(tensor_pc.shape[0], -1)

                # '''adapt layer'''
                z_2 = adapt_layer(z_1,obj = tensor_pc)


            recon_param = autoencoder.decoder(z_1+z_2)

            if args.mano =="oakink":
                '''ours'''
                recon_mano = rh_mano(recon_param[:, 10:58] , recon_param[:,:10])
                model_hand = recon_mano.verts.to(device) +  recon_param[:,None, 58:] # [B,778,3]
                '''gt'''
                # recon_param = torch.cat(gt_hand_param, dim=0)
                # recon_mano = rh_mano(recon_param[:, 10:58] , recon_param[:,:10])
                # model_hand = recon_mano.verts.to(device) +  recon_param[:,None, 58:] # [B,778,3]
            
            '''输出手部信息的 61维度信息'''
            for count, (tensor, hand) in enumerate(zip(recon_param, model_hand)):
                hand_out_list.append(hand.detach())

            '''cluster1'''
            kps = recon_mano.joints.to(device) + recon_param[:, None, 58:]
            for count, kps_i in enumerate(kps):
                cluster.append(kps_i.detach().reshape(-1).cpu().numpy())

            '''cluster2'''
            hand_kps = recon_mano.joints.to(device) + recon_param[:, None, 58:]
            is_right_vec = torch.ones(
                hand_kps.shape[0], device=hand_kps.device)

            hand_kps = convert_joints(
                hand_kps, source='mano', target='biomech')

            hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
            hand_kps_after = convert_joints(
                hand_kps_after, source='biomech', target='mano')

            for count, kps_flat in enumerate(hand_kps_after):
                cluster2.append(kps_flat.detach().reshape(-1).cpu().numpy())

            if flag < 4 :
                for x, (obj_face_1, obj_vert_1, model_hand_1, hand_face_1) in enumerate(zip(obj_face_list, obj_vert_list, hand_out_list, hand_face_list)):
                    vis_dataset(obj_face_1, obj_vert_1, model_hand_1.to("cpu"), hand_face_1.to("cpu"), f"{ply_path}/{flag * len(pc_list) + x}.ply")
            flag += 1
            '''dis volum '''
            simulation_displacement, penetration_distance, intersection_volume = grasp_displacement(obj_face_list, obj_vert_list, hand_out_list,  hand_face_list, "")

            simulation_displacements_list += simulation_displacement
            penetration_distances_list += penetration_distance
            intersection_volumes_list += intersection_volume 
            pc_list.clear()
            obj_face_list.clear()
            obj_vert_list.clear()
            hand_out_list.clear()
            hand_face_list.clear()  
            gt_hand_param.clear()
            th_v_template_list.clear()
            obj_mesh_path_list.clear()


        if args.dataset == "grab":
            th_v_template_list.append(th_v_template)

    cluster_array = np.array(cluster)
    entropy, cluster_size = diversity(cluster_array, cls_num=20)

    cluster_array_2 = np.array(cluster2)
    entropy_2, cluster_size_2 = diversity(cluster_array_2, cls_num=20)
    
    std_simulation_displacement = statistics.stdev(simulation_displacements_list)

    mean_simulation_displacement = sum(simulation_displacements_list) / len(simulation_displacements_list)
    
    mean_intersection_volume = sum(intersection_volumes_list) / len(intersection_volumes_list)
    
    mean_penetration_distance = sum(penetration_distances_list) / len(penetration_distances_list)
    contact_ratio= np.mean(np.array(intersection_volumes_list) != 0)


    with open(eval_path+f"val.txt", "a", encoding="utf-8") as file:
        print(f"guide_w : {args.guide_w}  , seed : {args.seed} , {args.sample}\n",file=file)
        print(f"mean_simulation_displacement : {mean_simulation_displacement * 1e2:.4f}e-02\n"
        f"std_simulation_displacement : {std_simulation_displacement * 1e2:.4f}e-02\n"
        f"mean_penetration_distance : {mean_penetration_distance * 1e2:.4f}e-02\n"
        f"mean_intersection_volume : {mean_intersection_volume * 1e6:.4f}e-06\n"
        f"contact_ratio : {contact_ratio * 1e2 :.4f}e-02\n"
        f"entropy :, {entropy} \n"
        f"cluster_size : {cluster_size}\n"
        f"entropy2 : {entropy_2} \n"
        f"cluster_size2 : {cluster_size_2}\n",file=file)


    print(f"guide_w : {args.guide_w} \n")
    print(f"mean_simulation_displacement : {mean_simulation_displacement * 1e2:.4f}e-02\n"
    f"std_simulation_displacement : {std_simulation_displacement * 1e2:.4f}e-02\n"
    f"mean_penetration_distance : {mean_penetration_distance * 1e2:.4f}e-02\n"
    f"mean_intersection_volume : {mean_intersection_volume * 1e6:.4f}e-06\n"
    f"contact_ratio : {contact_ratio * 1e2 :.4f}e-02\n"
    f"entropy :, {entropy} \n"
    f"cluster_size : {cluster_size}\n"
    f"entropy2 : {entropy_2} \n",
    f"cluster_size2 : {cluster_size_2} \n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config, 'r') as configfile:
        config = json.load(configfile)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()
    print(args)
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    # seed
    set_random_seed(args.seed)

    '''autoencoder'''
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
            new_key = key[7:]  # 去除前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    autoencoder.load_state_dict(new_state_dict)
    autoencoder = autoencoder.to(device)
    '''diffusion'''
    with open('config/diffusion.json', 'r') as f:
        diffusion_param = json.load(f)
    net = PointNet2SemSegSSG(diffusion_param)


    diffusion_model = DDIM(args , nn_model=net,
                           betas=(1e-4, 0.02), n_T=1000, device=device, drop_prob=0.1)

    checkpoint = torch.load(args.diffusion_path,
                            map_location=torch.device(device))
    new_state_dict = {}
    for key, value in checkpoint['network'].items():
        if key.startswith('module.'):
            new_key = key[7:]  # 去除前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    diffusion_model.load_state_dict(new_state_dict)
    diffusion_model = diffusion_model.to(device)
    '''adapt_Layer'''
    # adapt_layer = AdaptLayer(args.batch_size).to(device)
    adapt_layer = AdaptLayer2(args.adapt_layer["layer_sizes"], args.adapt_layer["latent_size"] , conditional= args.adapt_layer["conditional"] ,condition_size= args.adapt_layer["condition_size"]).to(device)
    # adapt_layer = AdaptLayer3(args.adapt_layer["layer_sizes"], args.adapt_layer["latent_size"] , conditional= args.adapt_layer["conditional"] ,condition_size= args.adapt_layer["condition_size"]).to(device)

    checkpoint = torch.load(args.adapt_path,map_location=torch.device(device))
    new_state_dict = {}
    for key, value in checkpoint['network'].items():
        if key.startswith('module.'):
            new_key = key[7:]  # 去除前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    adapt_layer.load_state_dict(new_state_dict)
    adapt_layer = adapt_layer.to(device)


    if 'Test' in args.train_mode:
        if args.dataset == "oakink":
            eval_dataset = Oakink(mode="test", batch_size=args.batch_size, args=args)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.dataloader_workers)
        else:
            eval_dataset = Grab(mode="test", batch_size=args.batch_size, args=args)
            desired_indices = [0 , 1 , 2 , 4 , 6 , 12]
            sampler = SubsetRandomSampler(desired_indices)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.dataloader_workers, sampler=sampler)
    # mano hand model
    with torch.no_grad():
        rh_mano = ManoLayer(
            center_idx=0, mano_assets_root="assets/mano_v1_2").to(device)
    # [1, 1538, 3], face triangle indexes
    rh_faces = rh_mano.th_faces.view(1, -1, 3).contiguous()
    rh_faces = rh_faces.repeat(
        args.batch_size, 1, 1).to(device)  # [N, 1538, 3]
    if 'Test' in args.train_mode:
        val(args, autoencoder, diffusion_model,adapt_layer,
            eval_loader, device, rh_mano, rh_faces, 'test')
    else:
        print("no dataset!")
