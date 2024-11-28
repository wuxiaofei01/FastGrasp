import argparse
import json
import os

import numpy as np
import trimesh
from joblib import Parallel, delayed

from evaluation.mano_train.simulation.simulate import process_sample
from evaluation.mano_train.netscripts.intersect import get_sample_intersect_volume
from evaluation.mano_train.networks.branches.contactloss import mesh_vert_int_exts
from tqdm import tqdm
import open3d as o3d
import torch
import time
import scipy
import scipy.cluster
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

def diversity(params_list, cls_num=20):
    # k-means
    params_list = scipy.cluster.vq.whiten(params_list)
    codes, dist = scipy.cluster.vq.kmeans(params_list, cls_num)  # codes: [20, 72], dist: scalar
    vecs, dist = scipy.cluster.vq.vq(params_list, codes)  # assign codes, vecs/dist: [1200]
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences  count: [20]
    ee = entropy(counts)
    return ee, np.mean(dist)

def batch_mesh_contains_points(ray_origins, obj_triangles,
                               direction=torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745])):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh
    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    tol_thresh: To determine if ray and triangle are //
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    tol_thresh = 0.0000001
    # ray_origins.requires_grad = False
    # obj_triangles.requires_grad = False
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    # Expand needed vectors
    batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3)

    # Compute ray/triangle intersections
    pvec = torch.cross(batch_direction, v0v2, dim=2)
    dets = torch.bmm(
        v0v1.view(batch_points_size, 1, 3), pvec.view(batch_points_size, 3, 1)
    ).view(batch_size, triangle_nb)

    # Check if ray and triangle are parallel
    parallel = abs(dets) < tol_thresh
    invdet = 1 / (dets + 0.1 * tol_thresh)

    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1)
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    )
    pvec = pvec.repeat(1, point_nb, 1)
    invdet = invdet.repeat(1, point_nb)
    tvec = hand_verts_repeated - v0
    u_val = (
        torch.bmm(
            tvec.view(batch_size * tvec.shape[1], 1, 3),
            pvec.view(batch_size * tvec.shape[1], 3, 1),
        ).view(batch_size, tvec.shape[1])
        * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    batch_direction = batch_direction.repeat(1, point_nb, 1)
    v_val = (
        torch.bmm(
            batch_direction.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
        torch.bmm(
            v0v2.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    parallel = parallel.repeat(1, point_nb)
    # # Check that all intersection conditions are met
    not_parallel = ~parallel
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0
    return exterior

def get_penetration(sample_info):
    obj_mesh = trimesh.Trimesh(vertices=sample_info["obj_verts"], faces=sample_info["obj_faces"])
    trimesh.repair.fix_normals(obj_mesh)

    obj_triangles = sample_info["obj_verts"][sample_info["obj_faces"]]
    exterior = batch_mesh_contains_points(torch.from_numpy(sample_info["hand_verts"][None, :, :]).float(),
                                                       torch.from_numpy(obj_triangles)[None, :, :, :].float())
    penetr_mask = ~exterior.squeeze(dim=0)

    if penetr_mask.sum() == 0:
        max_depth = 0
    else:
        (result_close, result_distance, _, ) = trimesh.proximity.closest_point(obj_mesh, sample_info["hand_verts"][penetr_mask == 1])
        max_depth = result_distance.max()

    return max_depth

def get_closed_faces(th_faces):
    close_faces = np.array(
        [
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ]
    )
    closed_faces = np.concatenate([th_faces, close_faces], axis=0)
    # Indices of faces added during closing --> should be ignored as they match the wrist
    # part of the hand, which is not an external surface of the human

    # Valid because added closed faces are at the end
    hand_ignore_faces = [1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]

    return closed_faces, hand_ignore_faces





def geneSampleInfos(fname_lists, hand_verts, hand_faces, object_verts, object_faces, scale=1):
    """
    Args:
        scale (float): convert to meters
    """
    # hand_faces = np.concatenate([expanded_array, expanded_array], axis=0)    
    # object_verts = torch.tensor(object_verts)
    # object_faces = torch.tensor(object_faces)
    # hand_verts = torch.tensor(hand_verts)
    sample_infos = []
    for hand_vert, hand_face, obj_vert, obj_face in tqdm(zip(
        hand_verts, hand_faces, object_verts, object_faces
    )):

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(obj_vert)
        mesh.triangles = o3d.utility.Vector3iVector(obj_face)
        mesh = mesh.simplify_quadric_decimation(50000)
        obj_vert = torch.from_numpy(np.asarray(mesh.vertices))
        obj_face = torch.from_numpy(np.asarray(mesh.triangles))

        hand_vert =hand_vert.to("cpu")
        hand_face = hand_face.to("cpu")
        obj_vert = obj_vert.to("cpu")
        obj_face = obj_face.to("cpu")

        hand_vert = hand_vert.numpy()
        hand_face = hand_face.numpy()
        obj_vert = obj_vert.numpy()
        obj_face = obj_face.numpy()
        sample_info = {
            "file_names": fname_lists,
            "hand_verts": hand_vert * scale,
            "hand_faces": hand_face,
            "obj_verts": obj_vert * scale,
            "obj_faces": obj_face,
        }
        obj_mesh = trimesh.load({"vertices": obj_vert, "faces": obj_face})
        trimesh.repair.fix_normals(obj_mesh)
        penetration = get_penetration(sample_info)
        sample_info["max_depth"] = penetration
        sample_infos.append(sample_info)

        

    return sample_infos

def para_simulate(
    sample_infos,  
    saved_path,
    wait_time=0,
    sample_vis_freq=100,
    use_gui=False,
    workers=8,
    cluster=False,
    vhacd_exe=None
):
    save_gif_folder = os.path.join(saved_path, "save_gifs")
    save_obj_folder = os.path.join(saved_path, "save_objs")
    # simulation_results_folder = os.path.join(saved_path,"simulation_results")
    # os.makedirs(save_gif_folder, exist_ok=True)
    # os.makedirs(save_obj_folder, exist_ok=True)
    # os.makedirs(simulation_results_folder, exist_ok=True)
    max_depths = [sample_info["max_depth"] for sample_info in sample_infos]
    # file_names = [sample_info["file_names"] for sample_info in sample_infos]
    distances = Parallel(n_jobs=workers)(
        delayed(process_sample)(
            sample_idx,
            sample_info,
            save_gif_folder=save_gif_folder,
            save_obj_folder=save_obj_folder,
            use_gui=use_gui,
            wait_time=wait_time,
            sample_vis_freq=sample_vis_freq,
            vhacd_exe=vhacd_exe,
        )
        for sample_idx, sample_info in enumerate(sample_infos)
    )

    volumes = Parallel(n_jobs=workers, verbose=5)(
        delayed(get_sample_intersect_volume)(sample_info)
        for sample_info in sample_infos
    )
    # simulation_results_path = os.path.join(simulation_results_folder,"results.json")
    # with open(simulation_results_path, "w") as j_f:
    #     json.dump(
    #         {
    #             "smi_dists": distances,#simulation displacement
    #             "mean_smi_dist": np.mean(distances),
    #             "std_smi_dist": np.std(distances),

    #             "max_depths": max_depths,  #penetration distance
    #             "mean_max_depth": np.mean(max_depths),

    #             "volumes": volumes, #intersection volume
    #             "mean_volume": np.mean(volumes),

    #             "file_names": file_names,

    #         },
    #         j_f,
    #     )
    #     print("Wrote results to {}".format(simulation_results_path))

    return distances , max_depths , volumes

def grasp_displacement(obj_faces , obj_verts , hand_verts , hand_faces ,file_names):



    vhacd_exe = "/public/home/v-wuxf/FastGrasp/testVHACD"

    sample_infos = geneSampleInfos(fname_lists=file_names,
                                   hand_verts=hand_verts,
                                   hand_faces=hand_faces,
                                   object_verts=obj_verts,
                                   object_faces=obj_faces)
    
    
    return para_simulate(sample_infos,file_names, vhacd_exe=vhacd_exe)



                                          