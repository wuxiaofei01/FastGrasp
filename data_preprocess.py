import argparse
import os
import numpy as np
from oikit.oi_shape import OakInkShape
from oikit.oi_shape.utils import viz_dataset
from termcolor import cprint
import trimesh
import ipdb
import torch
from tqdm import tqdm
import sys
from manotorch.manolayer import ManoLayer, MANOOutput
def vertices_transformation(vertices, rt):
    p = np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1)
    return p.T
def vis_dataset(obj_face , obj_vert,hand_vert,hand_face,filename):

    import open3d as o3d

    merged_point_cloud = o3d.geometry.PointCloud()
    
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_face)#画边
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_vert)#画顶点
    hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.array([[0.4, 0.81960784, 0.95294118]] * len(np.asarray(hand_vert))))
    hand_cloud = hand_mesh.sample_points_uniformly(number_of_points=100000)
    # hand_cloud = hand_cloud.farthest_point_down_sample(num_samples=4000)
    

    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_face)
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_vert)
    point_cloud = obj_mesh.sample_points_uniformly(number_of_points=100000)
    # point_cloud = point_cloud.farthest_point_down_sample(num_samples=4000)

    merged_point_cloud = point_cloud + hand_cloud
    # merged_point_cloud = hand_cloud

    o3d.io.write_point_cloud(filename, merged_point_cloud)
def main(arg):
    dataset = OakInkShape(category=arg.categories, intent_mode=arg.intent_mode, data_split=arg.data_split)
    obj_pc_list, hand_param_list = [], []
    obj_verts_list = []
    obj_faces_list = []
    for index in tqdm(range(len(dataset))):
        grasp = dataset[index]
        #原始数据
        verts = grasp["obj_verts"]# [314240,3]
        faces = grasp["obj_faces"]#(157088, 3)
        obj_verts_list.append(verts)
        obj_faces_list.append(faces)

        hand_shape = torch.tensor(grasp['hand_shape']).clone().detach()  # [10] 
        hand_rot = torch.tensor(grasp['hand_pose'][:3]).clone().detach()  # [3]
        hand_pose = torch.tensor(grasp['hand_pose'][3:]).clone().detach()  # [45] 
        hand_trans = torch.tensor(grasp['hand_tsl']).clone().detach()#[3]

        hand_param = torch.cat((hand_shape, hand_rot, hand_pose, hand_trans), dim=0)  # [61]
        hand_param_list.append(hand_param)

        #转换obj数据
        obj_mesh = trimesh.Trimesh(vertices=verts,faces=faces)
        obj_xyz_normalized, face_id = trimesh.sample.sample_surface(obj_mesh, 3000)
        obj_xyz_normalized = obj_xyz_normalized[:3000, :]  # [3000, 3]
        obj_pose = np.eye(4)
        obj_xyz_transformed = vertices_transformation(obj_xyz_normalized, obj_pose)
        obj_xyz_transformed = torch.tensor(obj_xyz_transformed, dtype=torch.float32)
        obj_scale = 1
        obj_scale_tensor = torch.tensor(obj_scale).type_as(obj_xyz_transformed).repeat(3000, 1)  # [3000, 1]
        obj_pc = torch.cat((obj_xyz_transformed, obj_scale_tensor), dim=-1)  # [3000, 4]
        obj_pc = obj_pc.permute(1, 0)  # [4, 3000]
        obj_pc_list.append(obj_pc)


        vis_dataset(faces, verts, grasp["verts"], dataset.mano_layer.th_faces, "1.ply")

        
        rh_mano = ManoLayer(center_idx=0, mano_assets_root="/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE/assets/mano_v1_2")
        hand_param = hand_param.unsqueeze(0)
        gt_mano = rh_mano(hand_param[:, 10:58] , hand_param[:,:10])
        gt_hand = gt_mano.verts +  hand_param[:,None, 58:] # [B,778,3]
        vis_dataset(faces, verts, gt_hand.squeeze(0), rh_mano.th_faces, "2.ply")
        exit()


    np.save(f"generate_data/obj_verts_{arg.data_split}.npy", obj_verts_list)

    np.save(f"generate_data/obj_faces_{arg.data_split}.npy", obj_faces_list)


    obj_pc_tensor = torch.stack(obj_pc_list, dim=0)  # [S, 4, 3000]
    obj_pc_tensor = obj_pc_tensor.cpu().numpy()
    np.save(f"generate_data/obj_pc_{arg.data_split}.npy", obj_pc_tensor)

    # save mano param
    hand_param_tensor = torch.stack(hand_param_list, dim=0)  # [S, 61]
    hand_param_tensor = hand_param_tensor.cpu().numpy()
    np.save(f"generate_data/hand_param_{arg.data_split}.npy",hand_param_tensor)

    print("EXIT")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OakInkImage sequence-level visualization")
    parser.add_argument("--data_dir", type=str, default="data", help="environment variable 'OAKINK_DIR'")
    parser.add_argument("--categories", type=str, default="all", help="list of object categories")
    parser.add_argument("--intent_mode", type=str, default="all", help="intent mode, list of intents")
    parser.add_argument("--data_split",
                        type=str,
                        default="val",
                        choices=["train", "test", "val"],
                        help="training data split")

    arg = parser.parse_args()
    os.environ["OAKINK_DIR"] = arg.data_dir
    main(arg)
