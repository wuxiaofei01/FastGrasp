from torch.utils.data import Dataset
import torch
import os
import pickle
from torchvision import transforms
import numpy as np
from utils import utils
import time
from PIL import Image
import json
from torch.utils import data
import trimesh
from manopth.manolayer import grabManoLayer
from manopth.rodrigues_layer import batch_rodrigues
from scipy.spatial.transform import Rotation

class Oakink():
    def __init__(self, mode="test", vis=False, batch_size=160,args = None):
        self.args = args
        self.mode = mode

        self.obj_pc_path = '/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE/hand-object-3/download/dataset/ObMan/obman/precessed/obj_pc_{}.npy'.format(mode)
        self.hand_param_path = '/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE/hand-object-3/download/dataset/ObMan/obman/precessed/hand_param_{}.npy'.format(mode)
        
        self.obj_faces_path = "/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE/hand-object-3/download/dataset/ObMan/obman/evaluation/obj_faces_{}.npy".format(mode)
        self.obj_verts_path = "/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE/hand-object-3/download/dataset/ObMan/obman/evaluation/obj_verts_{}.npy".format(mode)



        self.__load_dataset__()

        self.dataset_size = self.all_obj_pc.shape[0]

        self.transform = transforms.ToTensor()
        self.sample_nPoint = 3000
        self.batch_size = batch_size



    def __load_dataset__(self):
        print('loading dataset start')
        self.all_obj_pc = np.load(self.obj_pc_path)  # [S, 4, 3000]
        self.all_hand_param = np.load(self.hand_param_path)



        self.all_obj_faces = np.load(self.obj_faces_path, allow_pickle=True)
        self.all_obj_verts = np.load(self.obj_verts_path, allow_pickle=True)
        print('loading dataset finish')


    def __len__(self):
        return self.dataset_size 

    def __getitem__(self, idx):
        # obj_pc
        obj_pc = torch.tensor(self.all_obj_pc[idx], dtype = torch.float32)  # [4, 3000]
        # hand mano param
        hand_param = torch.tensor(self.all_hand_param[idx], dtype = torch.float32)  # [61]
        #obj_face
        obj_face = self.all_obj_faces[idx]
        #obj_vert
        obj_vert = self.all_obj_verts[idx]
        return {
            "obj_pc" : obj_pc,
            "hand_param" : hand_param,
            "obj_face" : torch.tensor(obj_face),
            "obj_vert":torch.tensor(obj_vert)
        }


class Grab(data.Dataset):
    def __init__(self, 
                 args,
                 dataset_root='grab_data',
                 mode='train',
                 n_samples=3000,
                 batch_size =256
                 ):
        super().__init__()
        self.mode = mode
        self.batch_size = batch_size
        self.ds_path = os.path.join(dataset_root, mode)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%mode))

        frame_names = np.load(os.path.join(dataset_root, mode, 'frame_names.npz'))['frame_names']
        self.frame_names = np.asarray([os.path.join(dataset_root, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])
        self.n_samples = n_samples
        self.sbjs = np.unique(self.frame_sbjs)
        self.sbj_info = np.load(os.path.join(dataset_root, 'sbj_info.npy'), allow_pickle=True).item()
        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
        
        with open(os.path.join("assets/closed_mano_faces.pkl"), 'rb') as f:
            self.hand_faces = pickle.load(f)

        self.obj_root = os.path.join(dataset_root, "obj_meshes")
        self.mano_layer = grabManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("assets/mano_v1_2/models"), use_pca=False, joint_rot_mode="rotmat")
        
    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        
    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]).float() for k in data.files}
        return data_torch

    def vertices_transformation(self,vertices, rt):
        p = np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1)
        return p.T

    def convert_rotmat_to_euler(self,rotmat):
        r = Rotation.from_matrix(rotmat.reshape(-1, 3, 3))
        euler = r.as_euler('xyz')
        euler = euler.reshape(rotmat.shape[:-2] + (3,))
        return euler
    def convert_euler_to_rotmat(self,euler):
        euler_list = [euler[i].tolist() for i in range(euler.shape[0])]
        
        rot_matrices = [Rotation.from_euler('xyz', angles).as_matrix() for angles in euler_list]

        rotmat_numpy = np.stack(rot_matrices)

        rotmat_torch = torch.tensor(rotmat_numpy, dtype=torch.float32)

        return rotmat_torch
    def __getitem__(self, item): 
        obj_name = self.frame_objs[item] #item = 54   , fframe_objs['cup', 'cylindersmall', 'train', 'spheresmall', 'stapler',...]  由于shuffle，打乱顺序
        obj_mesh_path = os.path.join(self.obj_root, obj_name + '.ply')# 'grab_data/obj_meshes/cubelarge.ply' 获取得到目标的ply文件位置
        obj_mesh = trimesh.load(obj_mesh_path, file_type="ply")# 获取得到mesh 
        obj_faces = obj_mesh.faces# 得到mesh的faces

        rot_mat = self.ds["root_orient_obj_rotmat"][item].numpy().reshape(3, 3)
        transl = self.ds["trans_obj"][item].numpy()
        obj_verts = obj_mesh.vertices @ rot_mat + transl # 获取得到 verts ，上述应该是数据增广？
        offset = obj_verts.mean(axis=0, keepdims=True)
        obj_verts = obj_verts - offset


        sbj_idx = self.frame_sbjs[item] # tensor(4)
        v_template = self.sbj_vtemp[sbj_idx] # [778,3] 这里应该是获取得到手的参数

        global_orient = self.ds['global_orient_rhand_rotmat'][item]
        rhand_rotmat = self.ds['fpose_rhand_rotmat'][item]

        handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)# 获取方位信息

        th_trans = self.ds['trans_rhand'][item].unsqueeze(dim=0) - torch.FloatTensor(offset)
        th_v_template = v_template.unsqueeze(dim=0)

        hand_verts, hand_frames,_ = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)  # 需要知道handrot,th_trans,th_v_template

        obj_verts = obj_verts @ rot_mat.T
        global_orient = torch.from_numpy(rot_mat).float() @ global_orient
        handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)
        
        root_center = hand_frames[:, 0, :3, 3]
        th_trans = (root_center[:, None, :] @ rot_mat.T).squeeze(dim=1) - root_center + th_trans

        hand_verts, hand_frames,_ = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)

        hand_verts = hand_verts.squeeze(dim=0).float()

        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
        
        # import ipdb
        # import open3d as o3d
        # hand_mesh = o3d.geometry.TriangleMesh()
        # hand_mesh.triangles = o3d.utility.Vector3iVector(self.hand_faces)#画边
        # hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)#画顶点
        # hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
        #     np.array([[0.4, 0.81960784, 0.95294118]] * len(np.asarray(hand_verts))))

        # hand_cloud = hand_mesh.sample_points_uniformly(number_of_points=100000)

        # o3d.io.write_point_cloud("hand_0.ply", hand_cloud)
        
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

        '''
        将grab数据集转换为param51 , rot - > param
        handrot : [ 16 , 3 , 3]
        th_trans : [3]
        只需要找到 旋转矩阵和旋转角之间的映射关系就行
        '''
        handrot_param48 = torch.tensor(self.convert_rotmat_to_euler(handrot.to("cpu")) , dtype = torch.float32).reshape(1, -1)   # [B , 48]  totmat -> euler
        hand_param = torch.cat((handrot_param48 , th_trans),dim = -1).squeeze(0)

        '''测试使用矩阵逆运算,精度是否能接受.通过可视化查看'''

        # hand_param = torch.cat((handrot_param48 , th_trans),dim = -1)
        # handeuler = hand_param[:, :48].view(-1,16,3)  # [ B , 16 , 3 ]
        # handrot = self.convert_euler_to_rotmat(handeuler)# [ B , 16 , 3 , 3]
        # th_trans =hand_param[:, 48:] # [ B , 3]
        # hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)

        '''
        查看逆运算是否成功
        由于精度误差关系,导致有些出入.
        '''
        # recon_handrot = self.convert_euler_to_rotmat(handrot_param48.reshape(1,16,3)) # [1 , 16 , 3 , 3]
        # difference = torch.abs(recon_handrot - handrot)
        # tolerance = 5e6
        # if difference.max() < tolerance:
        #     print("The two tensors are effectively the same.")
        # else:
        #     print("The two tensors are not exactly the same.")
        # exit()



        """可视化手 + 物体
        import ipdb
        import open3d as o3d
        ipdb.set_trace() # 可视化物体点云


        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(obj_verts)
        o3d.io.write_point_cloud("obj.ply", point_cloud)
        hand_verts = hand_verts.squeeze(0)

        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.triangles = o3d.utility.Vector3iVector(self.hand_faces)#画边
        hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)#画顶点
        hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([[0.4, 0.81960784, 0.95294118]] * len(np.asarray(hand_verts))))

        hand_cloud = hand_mesh.sample_points_uniformly(number_of_points=100000)

        o3d.io.write_point_cloud("hand_1.ply", hand_cloud)
        """
        
        return {
            "obj_pc": obj_pc,
            "hand_param": hand_param,
            "hand_verts": hand_verts,
            "th_v_template":th_v_template.squeeze(0),
            "hand_faces" : self.hand_faces,
            "obj_face" : torch.tensor(obj_faces).squeeze(0),
            "obj_vert":torch.tensor(obj_verts).squeeze(0),
            "obj_mesh_path":obj_mesh_path
        }




class GrabTest(data.Dataset):
    def __init__(self, 
                 args,
                 dataset_root='grab_data',
                 mode='train',
                 n_samples=3000,
                 batch_size =256
                 ):
        super().__init__()
        self.mode = mode
        self.batch_size = batch_size
        self.ds_path = os.path.join(dataset_root, mode)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%mode))

        frame_names = np.load(os.path.join(dataset_root, mode, 'frame_names.npz'))['frame_names']
        self.frame_names = np.asarray([os.path.join(dataset_root, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])
        self.n_samples = n_samples
        self.sbjs = np.unique(self.frame_sbjs)
        self.sbj_info = np.load(os.path.join(dataset_root, 'sbj_info.npy'), allow_pickle=True).item()
        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
        
        with open(os.path.join("assets/closed_mano_faces.pkl"), 'rb') as f:
            self.hand_faces = pickle.load(f)

        self.obj_root = os.path.join(dataset_root, "obj_meshes")
        self.mano_layer = grabManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("assets/mano_v1_2/models"), use_pca=False, joint_rot_mode="rotmat")
        
    def __len__(self):
        return 10
        
    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]).float() for k in data.files}
        return data_torch

    def vertices_transformation(self,vertices, rt):
        p = np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1)
        return p.T

    def convert_rotmat_to_euler(self,rotmat):
        r = Rotation.from_matrix(rotmat.reshape(-1, 3, 3))
        euler = r.as_euler('xyz')
        euler = euler.reshape(rotmat.shape[:-2] + (3,))
        return euler
    def convert_euler_to_rotmat(self,euler):
        euler_list = [euler[i].tolist() for i in range(euler.shape[0])]
        
        rot_matrices = [Rotation.from_euler('xyz', angles).as_matrix() for angles in euler_list]

        rotmat_numpy = np.stack(rot_matrices)

        rotmat_torch = torch.tensor(rotmat_numpy, dtype=torch.float32)

        return rotmat_torch
    def __getitem__(self, item): 
        obj_name = self.frame_objs[item] #item = 54   , fframe_objs['cup', 'cylindersmall', 'train', 'spheresmall', 'stapler',...]  由于shuffle，打乱顺序
        obj_mesh_path = os.path.join(self.obj_root, obj_name + '.ply')# 'grab_data/obj_meshes/cubelarge.ply' 获取得到目标的ply文件位置
        obj_mesh = trimesh.load(obj_mesh_path, file_type="ply")# 获取得到mesh 
        obj_faces = obj_mesh.faces# 得到mesh的faces

        rot_mat = self.ds["root_orient_obj_rotmat"][item].numpy().reshape(3, 3)
        transl = self.ds["trans_obj"][item].numpy()
        obj_verts = obj_mesh.vertices @ rot_mat + transl # 获取得到 verts ，上述应该是数据增广？
        offset = obj_verts.mean(axis=0, keepdims=True)
        obj_verts = obj_verts - offset


        sbj_idx = self.frame_sbjs[item] # tensor(4)
        v_template = self.sbj_vtemp[sbj_idx] # [778,3] 这里应该是获取得到手的参数

        global_orient = self.ds['global_orient_rhand_rotmat'][item]
        rhand_rotmat = self.ds['fpose_rhand_rotmat'][item]

        handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)# 获取方位信息

        th_trans = self.ds['trans_rhand'][item].unsqueeze(dim=0) - torch.FloatTensor(offset)
        th_v_template = v_template.unsqueeze(dim=0)

        hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)  # 需要知道handrot,th_trans,th_v_template

        obj_verts = obj_verts @ rot_mat.T
        global_orient = torch.from_numpy(rot_mat).float() @ global_orient
        handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)
        
        root_center = hand_frames[:, 0, :3, 3]
        th_trans = (root_center[:, None, :] @ rot_mat.T).squeeze(dim=1) - root_center + th_trans

        hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)
        if self.mode == "train":
            orient = torch.FloatTensor(1, 3).uniform_(-np.pi/6, np.pi/6)
            aug_rot_mats = batch_rodrigues(orient.view(-1, 3)).view([1, 3, 3])
            aug_rot_mat = aug_rot_mats[0]
            obj_verts = obj_verts @ aug_rot_mat.numpy().T
            global_orient = aug_rot_mat @ global_orient
            handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)

            root_center = hand_frames[:, 0, :3, 3]
            th_trans = (root_center[:, None, :] @ aug_rot_mat.T).squeeze(dim=1) - root_center + th_trans
            hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)

        hand_verts = hand_verts.squeeze(dim=0).float()

        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
        
        # import ipdb
        # import open3d as o3d
        # hand_mesh = o3d.geometry.TriangleMesh()
        # hand_mesh.triangles = o3d.utility.Vector3iVector(self.hand_faces)#画边
        # hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)#画顶点
        # hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
        #     np.array([[0.4, 0.81960784, 0.95294118]] * len(np.asarray(hand_verts))))

        # hand_cloud = hand_mesh.sample_points_uniformly(number_of_points=100000)

        # o3d.io.write_point_cloud("hand_0.ply", hand_cloud)
        
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

        '''
        将grab数据集转换为param51 , rot - > param
        handrot : [ 16 , 3 , 3]
        th_trans : [3]
        只需要找到 旋转矩阵和旋转角之间的映射关系就行
        '''
        handrot_param48 = torch.tensor(self.convert_rotmat_to_euler(handrot.to("cpu")) , dtype = torch.float32).reshape(1, -1)   # [B , 48]  totmat -> euler
        hand_param = torch.cat((handrot_param48 , th_trans),dim = -1).squeeze(0)

        '''测试使用矩阵逆运算,精度是否能接受.通过可视化查看'''

        # hand_param = torch.cat((handrot_param48 , th_trans),dim = -1)
        # handeuler = hand_param[:, :48].view(-1,16,3)  # [ B , 16 , 3 ]
        # handrot = self.convert_euler_to_rotmat(handeuler)# [ B , 16 , 3 , 3]
        # th_trans =hand_param[:, 48:] # [ B , 3]
        # hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)

        '''
        查看逆运算是否成功
        由于精度误差关系,导致有些出入.
        '''
        # recon_handrot = self.convert_euler_to_rotmat(handrot_param48.reshape(1,16,3)) # [1 , 16 , 3 , 3]
        # difference = torch.abs(recon_handrot - handrot)
        # tolerance = 5e6
        # if difference.max() < tolerance:
        #     print("The two tensors are effectively the same.")
        # else:
        #     print("The two tensors are not exactly the same.")
        # exit()



        """可视化手 + 物体
        import ipdb
        import open3d as o3d
        ipdb.set_trace() # 可视化物体点云


        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(obj_verts)
        o3d.io.write_point_cloud("obj.ply", point_cloud)
        hand_verts = hand_verts.squeeze(0)

        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.triangles = o3d.utility.Vector3iVector(self.hand_faces)#画边
        hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)#画顶点
        hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([[0.4, 0.81960784, 0.95294118]] * len(np.asarray(hand_verts))))

        hand_cloud = hand_mesh.sample_points_uniformly(number_of_points=100000)

        o3d.io.write_point_cloud("hand_1.ply", hand_cloud)
        """
        
        return {
            "obj_pc": obj_pc,
            "hand_param": hand_param,
            "hand_verts": hand_verts,
            "th_v_template":th_v_template.squeeze(0),
            "hand_faces" : self.hand_faces,
            "obj_face" : torch.tensor(obj_faces).squeeze(0),
            "obj_vert":torch.tensor(obj_verts).squeeze(0),
            "obj_mesh_path":obj_mesh_path
        }
