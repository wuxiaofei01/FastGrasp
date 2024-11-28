import numpy as np
import torch
import os

def vis_dataset(obj_face , obj_vert,hand_vert,hand_face,filename):

    import open3d as o3d

    merged_point_cloud = o3d.geometry.PointCloud()
    
    obj_face.to("cpu")
    obj_vert.to("cpu")

    hand_vert = hand_vert.squeeze(dim=0)
    hand_vert = hand_vert.numpy()

    obj_face = obj_face.squeeze(dim = 0)
    obj_face = obj_face.numpy()
    obj_vert = obj_vert.squeeze(dim = 0)
    obj_vert = obj_vert.numpy()
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_face)#画边
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_vert)#画顶点
    hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.array([[0.4, 0.81960784, 0.95294118]] * len(np.asarray(hand_vert))))
    hand_cloud = hand_mesh.sample_points_uniformly(number_of_points=100000)
    

    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_face)
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_vert)
    point_cloud = obj_mesh.sample_points_uniformly(number_of_points=100000)

    """save as point cloud"""
    # merged_point_cloud = point_cloud + hand_cloud
    # o3d.io.write_point_cloud(filename, merged_point_cloud)

    combined_mesh = hand_mesh + obj_mesh
    o3d.io.write_triangle_mesh(filename, combined_mesh)

def vis_dataset2(obj_face , obj_vert,hand_vert,hand_face,filename):

    import open3d as o3d

    merged_point_cloud = o3d.geometry.PointCloud()
    
    obj_face.to("cpu")
    obj_vert.to("cpu")

    hand_vert = hand_vert.squeeze(dim=0)
    hand_vert = hand_vert.numpy()

    obj_face = obj_face.squeeze(dim = 0)
    obj_face = obj_face.numpy()
    obj_vert = obj_vert.squeeze(dim = 0)
    obj_vert = obj_vert.numpy()
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
    

    directory, filename = os.path.split(filename)
    filename_without_ext, ext = os.path.splitext(filename)

    new_directory = os.path.join(directory, filename_without_ext)

    if  not os.path.exists(new_directory):#如果路径不存在

        os.makedirs(new_directory)
    hand_path = os.path.join(new_directory, "hand.ply")
    obj_path = os.path.join(new_directory, "object.ply")
    fuse_path = os.path.join(new_directory, "fuse.ply")
    hand_point_path = os.path.join(new_directory, "hand_point.ply")
    hand_cloud = hand_mesh.sample_points_uniformly(number_of_points=100000)

    obj_point_path = os.path.join(new_directory, "obj_point.ply")
    obj_cloud = obj_mesh.sample_points_uniformly(number_of_points=100000)

    o3d.io.write_triangle_mesh(obj_path, obj_mesh)

    o3d.io.write_triangle_mesh(hand_path, hand_mesh)

    o3d.io.write_triangle_mesh(fuse_path, obj_mesh + hand_mesh)

    o3d.io.write_point_cloud(hand_point_path, hand_cloud)
    o3d.io.write_point_cloud(obj_point_path, obj_cloud)

def vis_hand(hand_param , filename):
    import open3d as o3d
    from manotorch.manolayer import ManoLayer, MANOOutput
    hand_param = torch.tensor(hand_param).to("cpu")

    rh_mano = ManoLayer(center_idx=0, mano_assets_root="/public/home/v-wuxf/CVPR/GraspTTA/affordance-CVAE/assets/mano_v1_2")

    hand_param = hand_param.unsqueeze(0)
    gt_mano = rh_mano(hand_param[:,10:58] , hand_param[:,:10])
    gt_hand = gt_mano.verts +  hand_param[:,58:] # [B,778,3]
    gt_hand = gt_hand.squeeze(0)

    merged_point_cloud = o3d.geometry.PointCloud()

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(rh_mano.th_faces)#画边
    hand_mesh.vertices = o3d.utility.Vector3dVector(gt_hand)#画顶点
    hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.array([[0.4, 0.81960784, 0.95294118]] * len(np.asarray(gt_hand))))

    hand_cloud = hand_mesh.sample_points_uniformly(number_of_points=100000)

    o3d.io.write_point_cloud(filename, hand_cloud)