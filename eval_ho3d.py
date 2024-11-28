import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from dataset.HO3D_diversity_generation import HO3D_diversity
from manotorch.manolayer import ManoLayer, MANOOutput
from network.affordanceNet_obman_mano_vertex import affordanceNet
from network.autoencoder.autoencoder import Autoencoder
from network.adapt_layer.adapt_layer import AdaptLayer, AdaptLayer2, AdaptLayer3
from network.diffusion.ddim import DDIM
from network.diffusion.pointnet2.pointnet2_ssg_sem import PointNet2SemSegSSG
import numpy as np
import random
from utils import utils, utils_loss
import json
from evaluation.displacement import grasp_displacement, diversity
import trimesh
from tqdm import tqdm
from metric.simulate import run_simulation
from evaluation.vis import vis_dataset
from evaluation.converter import transform_to_canonical , convert_joints
# from scipy.spatial.transform import Rotation
import shutil
from tqdm import tqdm
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign


def convert_euler_to_rotmat(euler):
    euler_list = [euler[i].tolist() for i in range(euler.shape[0])]
    rot_matrices = [Rotation.from_euler(
        'xyz', angles).as_matrix() for angles in euler_list]
    rotmat_numpy = np.stack(rot_matrices)
    rotmat_torch = torch.tensor(rotmat_numpy, dtype=torch.float32)
    return rotmat_torch

def convert_rotmat_to_euler(rotmat):
    r = Rotation.from_matrix(rotmat.reshape(-1, 3, 3))
    euler = r.as_euler('xyz')
    euler = euler.reshape(rotmat.shape[:-2] + (3,))
    return euler
def convert_euler_to_rotmat(euler):
    euler_list = [euler[i].tolist() for i in range(euler.shape[0])]
    
    rot_matrices = [Rotation.from_euler('xyz', angles).as_matrix() for angles in euler_list]

    rotmat_numpy = np.stack(rot_matrices)

    rotmat_torch = torch.tensor(rotmat_numpy, dtype=torch.float32)

    return rotmat_torch

def main(args, autoencoder, diffusion_model, adapt_layer, eval_loader, device, rh_mano, rh_faces):
    '''
    Generate diverse grasps for object index with args.obj_id in out-of-domain HO3D object models
    '''
    guide_w = args.guide_w

    autoencoder.eval()
    diffusion_model.eval()
    adapt_layer.eval()
    penetr_vol_list, simu_disp_list, sample_contact_list = [], [], []
    cluster = []
    cluster2 = []
    kps_list_npy = []
    model_path = f"{os.path.dirname(args.diffusion_path)}"
    eval_path = model_path + "/ho3d/"
    ply_path = model_path + "/ho3d_ply"
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    if not os.path.exists(ply_path):
        os.makedirs(ply_path)
    execution_time = 0
    with torch.no_grad():
        rh_mano.eval()
        for batch_idx, (obj_id, obj_pc, origin_verts, origin_faces) in tqdm(enumerate(eval_loader)):
            if obj_id.item() not in args.obj_id:
                continue
            obj_xyz = obj_pc.permute(0, 2, 1)[:, :, :3].squeeze(0).cpu().numpy()  # [3000, 3]
            origin_verts = origin_verts.squeeze(0).numpy()  # [N, 3]
            penetr_vol_list_1, simu_disp_list_1, sample_contact_list_1 = [], [], []

            start_time =time.time()
            z_1 = diffusion_model.sample_ddim(10, (256, 3), device, guide_w=guide_w ,ddim_step= args.ddim_step,obj_feature=obj_pc.repeat(10,1,1).to("cuda") ).view(10, -1)

            """adapt layer"""
            z_2 = adapt_layer(z_1 , obj =obj_pc.repeat(10,1,1).to("cuda"))
            recon_param = autoencoder.decoder(z_1+z_2)
            """diffusion"""

            # recon_param = autoencoder.decoder(z_1)

            end_time = time.time()  # 记录结束时间
            execution_time = end_time - start_time + execution_time # 计算执行时间

            obj_mesh = trimesh.Trimesh(vertices=origin_verts, faces=origin_faces.squeeze(
                0).cpu().numpy().astype(np.int32))  # obj
            recon_mano = rh_mano(recon_param[:, 10:58], recon_param[:, :10])
            final_mano_verts = recon_mano.verts.to(device) + recon_param[:, None, 58:]  # [B,778,3]



            for i, final_mano_vert in tqdm(enumerate(final_mano_verts)):
                vis_dataset(origin_faces, torch.from_numpy(origin_verts).unsqueeze(0), final_mano_vert.to("cpu"), rh_mano.th_faces.cpu().numpy(), f"{ply_path}/{batch_idx}_{i}.ply")

                
                try:
                    hand_mesh = trimesh.Trimesh(vertices=final_mano_vert.to("cpu").squeeze(
                        dim=0).numpy(), faces=rh_mano.th_faces.cpu().numpy())
                except:
                    continue

                '''TTA'''
                # penetration volume
                penetr_vol = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
                # contact
                penetration_tol = 0.005
                result_close, result_distance, _ = trimesh.proximity.closest_point(
                    obj_mesh, final_mano_vert.to("cpu").squeeze(dim=0).numpy())
                sign = mesh_vert_int_exts(
                    obj_mesh, final_mano_vert.to("cpu").squeeze(dim=0).numpy())
                nonzero = result_distance > penetration_tol
                exterior = [sign == -1][0] & nonzero
                contact = ~exterior
                sample_contact = contact.sum() > 0
                # simulation displacement
                vhacd_exe = "/public/home/v-wuxf/FastGrasp/testVHACD"
                try:
                    simu_disp = run_simulation(final_mano_vert.to("cpu").squeeze(dim=0).numpy(), rh_mano.th_faces.cpu().numpy(),
                                               origin_verts, origin_faces.cpu().numpy().astype(np.int32).reshape((-1, 3)),
                                               vhacd_exe=vhacd_exe, sample_idx=batch_idx)
                except:
                    simu_disp = 0.10
                    
                penetr_vol_list_1.append(penetr_vol)
                simu_disp_list_1.append(simu_disp)
                sample_contact_list_1.append(sample_contact)



            kps = recon_mano.joints.to(device) + recon_param[:, None, 58:] 

            for kps_idx in kps:
                kps_list_npy.append(kps_idx.detach().cpu().numpy())


            for count, kps_flat in enumerate(kps):
                cluster.append(kps_flat.reshape(-1).cpu().numpy())
            

            hand_kps = recon_mano.joints.to(device) + recon_param[:, None, 58:] 
            hand_kps = hand_kps * 100
            is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)

            hand_kps = convert_joints(hand_kps, source='mano', target='biomech')

            hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
            hand_kps_after = convert_joints(hand_kps_after, source='biomech', target='mano')

            for count, kps_flat in enumerate(hand_kps_after):
                cluster2.append(kps_flat.reshape(-1).cpu().numpy())

            penetr_vol_list += penetr_vol_list_1
            simu_disp_list += simu_disp_list_1
            sample_contact_list += sample_contact_list_1
            
        # np.save("ho3d.npy",kps_list_npy)
        cluster_array = np.array(cluster)
        entropy, cluster_size = diversity(cluster_array, cls_num=20)

        cluster_array_2 = np.array(cluster2)
        entropy_2, cluster_size_2 = diversity(cluster_array_2, cls_num=20)
        print(execution_time)
        '''TTA'''
        with open(eval_path+f"val.log", "a", encoding="utf-8") as file:
            print(args.diffusion_path, file=file)
            print(f"ddim - {args.guide_w} -- {args.seed} -- {args.ddim_step} \n" ,  file=file)

            print(f"mean_simulation_displacement : {np.mean(simu_disp_list) * 1e2:.4f}e-02\n"
                  f"mean_intersection_volume : {np.mean(penetr_vol_list) * 1e6:.4f}e-06\n"
                  f"contact_ratio : {np.mean(sample_contact_list) * 1e2 :.4f}e-02\n"
                  f"entropy :, {entropy}\n"
                  f"cluster_size : {cluster_size}\n",
                  f"entropy_2 : {entropy_2} \n"
                  f"cluster_size_2 : {cluster_size_2}", file=file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="config/diff/oakink/ho3d.json",
                        help='Path to the JSON config file')

    args = parser.parse_args()
    
    with open(args.config, 'r') as configfile:
        config = json.load(configfile)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()
    # device
    set_random_seed(args.seed)
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(args)
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


    diffusion_model = DDIM(args,nn_model=net,betas=(1e-4, 0.02), n_T=1000, device=device, drop_prob=0.1)

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
    adapt_layer = AdaptLayer2(args.adapt_layer["layer_sizes"], args.adapt_layer["latent_size"],
                              conditional=args.adapt_layer["conditional"], condition_size=args.adapt_layer["condition_size"]).to(device)
    # adapt_layer = AdaptLayer3(args.adapt_layer["layer_sizes"], args.adapt_layer["latent_size"] , conditional= args.adapt_layer["conditional"] ,condition_size= args.adapt_layer["condition_size"]).to(device)

    checkpoint = torch.load(args.adapt_path, map_location=torch.device(device))
    new_state_dict = {}
    for key, value in checkpoint['network'].items():
        if key.startswith('module.'):
            new_key = key[7:]  # 去除前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    adapt_layer.load_state_dict(new_state_dict)
    adapt_layer = adapt_layer.to(device)

    """
    cmap_model = pointnet_reg(with_rgb=False)  # ContactNet

    checkpoint_cmap = torch.load(args.cmap_model_path, map_location=torch.device('cpu'))['network']
    cmap_model.load_state_dict(checkpoint_cmap)
    cmap_model = cmap_model.to(device)
    """

    # dataset
    dataset = HO3D_diversity()
    dataloader = DataLoader(dataset=dataset, batch_size=1,
                            shuffle=False, num_workers=4)
    # mano hand model
    with torch.no_grad():
        rh_mano = ManoLayer(
            center_idx=0, mano_assets_root="/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE/assets/mano_v1_2").to(device)
    rh_faces = rh_mano.th_faces.view(1, -1, 3).contiguous()

    main(args, autoencoder, diffusion_model, adapt_layer,
         dataloader, device, rh_mano, rh_faces)
