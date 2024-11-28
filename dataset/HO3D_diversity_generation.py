import os
import numpy as np
import random
from PIL import ImageFilter
from PIL import Image
from torch.utils import data
from dataset import utils_HO3D_FPHA
import torch
import cv2
import pickle
from utils import utils
from torch.utils.data import DataLoader

import trimesh
class HO3D_diversity(data.Dataset):
    def __init__(self):
        self.obj_root = 'dataset/HO3D_Object_models'

        obj_pc_dict, obj_face_dict, obj_scale_dict, obj_pc_resample_dict, obj_resample_faceid_dict = utils_HO3D_FPHA.load_objects_HO3D(self.obj_root)
        self.obj_pc_dict = obj_pc_dict
        self.obj_face_dict = obj_face_dict
        self.obj_scale_dict = obj_scale_dict
        self.obj_pc_resample_dict = obj_pc_resample_dict
        self.obj_resample_faceid_dict = obj_resample_faceid_dict
        self.nPoint = 3000

        self.obj_list = list(self.obj_pc_dict.keys())

    def __len__(self):
        return len(self.obj_list)
    def vertices_transformation(self,vertices, rt):
        p = np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1)
        return p.T
    def __getitem__(self, idx):
        obj_name = self.obj_list[idx]
        obj_id = torch.tensor(int(obj_name[:3]), dtype=torch.float32)

        origin_verts = torch.tensor(self.obj_pc_dict[obj_name], dtype=torch.float32)
        origin_faces = torch.tensor(self.obj_face_dict[obj_name], dtype=torch.float32)

        verts = torch.tensor(self.obj_pc_resample_dict[obj_name][:self.nPoint, :], dtype=torch.float32)  # [3000, 3]
        scale = self.obj_scale_dict[obj_name]
        obj_scale_tensor = torch.tensor(scale).type_as(verts).repeat(self.nPoint, 1)  # [3000, 1]
        # obj_pc = torch.cat((verts, obj_scale_tensor), dim=-1)  # [N', 4]
        # obj_pc = obj_pc.permute(1, 0)  # [4, N']

        obj_mesh = trimesh.Trimesh(vertices=origin_verts, faces=origin_faces)
        
        """获取得到obj_pc"""
        obj_xyz_normalized, face_id = trimesh.sample.sample_surface(obj_mesh, 3000)
        obj_xyz_normalized = obj_xyz_normalized[:3000, :]  # [3000, 3]
        obj_pose = np.eye(4)
        obj_xyz_transformed = self.vertices_transformation(obj_xyz_normalized, obj_pose)
        obj_xyz_transformed = torch.tensor(obj_xyz_transformed, dtype=torch.float32)
        obj_scale = 1
        obj_scale_tensor = torch.tensor(obj_scale).type_as(obj_xyz_transformed).repeat(3000, 1)  # [3000, 1]
        obj_pc = torch.cat((obj_xyz_transformed, obj_scale_tensor), dim=-1)  # [3000, 4]
        obj_pc = obj_pc.permute(1, 0)  # [4, 3000]

        return obj_id, obj_pc, origin_verts, origin_faces


if __name__ == '__main__':
    dataset = HO3D_diversity()
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    for idx, (obj_id, obj_pc, origin_obj_xyz, origin_obj_faces) in enumerate(dataloader):
        print(obj_id)