import torch

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)

# customized
import sys
sys.path.append(".")

from lib.constants import VIEWPOINTS

# ---------------- UTILS ----------------------

def degree_to_radian(d):
    return d * np.pi / 180

def radian_to_degree(r):
    return 180 * r / np.pi

def xyz_to_polar(xyz):
    """ assume y-axis is the up axis """
    
    x, y, z = xyz
    
    theta = 180 * np.arccos(z) / np.pi
    phi = 180 * np.arccos(y) / np.pi

    return theta, phi

def polar_to_xyz(theta, phi, dist):
    """ assume y-axis is the up axis """

    theta = degree_to_radian(theta)
    phi = degree_to_radian(phi)

    x = np.sin(phi) * np.sin(theta) * dist
    y = np.cos(phi) * dist
    z = np.sin(phi) * np.cos(theta) * dist

    return [x, y, z]


# ---------------- VIEWPOINTS ----------------------


def filter_viewpoints(pre_viewpoints: dict, viewpoints: dict):
    """ return the binary mask of viewpoints to be filtered """

    filter_mask = [0 for _ in viewpoints.keys()]
    for i, v in viewpoints.items():
        x_v, y_v, z_v = polar_to_xyz(v["azim"], 90 - v["elev"], v["dist"])

        for _, pv in pre_viewpoints.items():
            x_pv, y_pv, z_pv = polar_to_xyz(pv["azim"], 90 - pv["elev"], pv["dist"])
            sim = cosine_similarity(
                np.array([[x_v, y_v, z_v]]),
                np.array([[x_pv, y_pv, z_pv]])
            )[0, 0]

            if sim > 0.9:
                filter_mask[i] = 1

    return filter_mask


def init_viewpoints(mode, sample_space, init_dist, init_elev, principle_directions, 
    use_principle=True, use_shapenet=False, use_objaverse=False):

    if mode == "predefined":

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list
        ) = init_predefined_viewpoints(sample_space, init_dist, init_elev)

    elif mode == "hemisphere":

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list
        ) = init_hemisphere_viewpoints(sample_space, init_dist)

    else:
        raise NotImplementedError()

    # punishments for views -> in case always selecting the same view
    view_punishments = [1 for _ in range(len(dist_list))]

    if use_principle:

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments,
            length
        ) = init_principle_viewpoints(
            principle_directions, 
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments,
            use_shapenet,
            use_objaverse
        )

    return dist_list, elev_list, azim_list, sector_list, view_punishments, length


def init_principle_viewpoints(
    principle_directions, 
    dist_list, 
    elev_list, 
    azim_list, 
    sector_list,
    view_punishments,
    use_shapenet=False,
    use_objaverse=False
):

    if use_shapenet:
        key = "shapenet"

        pre_elev_list = [v for v in VIEWPOINTS[key]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[key]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[key]["sector"]]

        num_principle = 10
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    elif use_objaverse:
        key = "objaverse"

        pre_elev_list = [v for v in VIEWPOINTS[key]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[key]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[key]["sector"]]

        num_principle = len(pre_azim_list)
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]
    else:
        num_principle = 6
        pre_elev_list = [v for v in VIEWPOINTS[num_principle]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[num_principle]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[num_principle]["sector"]]
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    dist_list = pre_dist_list + dist_list
    elev_list = pre_elev_list + elev_list
    azim_list = pre_azim_list + azim_list
    sector_list = pre_sector_list + sector_list
    view_punishments = pre_view_punishments + view_punishments

    return dist_list, elev_list, azim_list, sector_list, view_punishments, len(pre_azim_list)


def init_predefined_viewpoints(sample_space, init_dist, init_elev):
    
    viewpoints = VIEWPOINTS[sample_space]

    assert sample_space == len(viewpoints["sector"])

    dist_list = [init_dist for _ in range(sample_space)] # always the same dist
    elev_list = [viewpoints["elev"][i] for i in range(sample_space)]
    azim_list = [viewpoints["azim"][i] for i in range(sample_space)]
    sector_list = [viewpoints["sector"][i] for i in range(sample_space)]
    return dist_list, elev_list, azim_list, sector_list


def init_hemisphere_viewpoints(sample_space, init_dist):
    """
        y is up-axis
    """

    num_points = 2 * sample_space
    ga = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    flags = []
    elev_list = [] # degree
    azim_list = [] # degree

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1

        # only take the north hemisphere
        if y >= 0: 
            flags.append(True)
        else:
            flags.append(False)

        theta = ga * i  # golden angle increment

        elev_list.append(radian_to_degree(np.arcsin(y)))
        azim_list.append(radian_to_degree(theta))

        radius = np.sqrt(1 - y * y)  # radius at y
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

    elev_list = [elev_list[i] for i in range(len(elev_list)) if flags[i]]
    azim_list = [azim_list[i] for i in range(len(azim_list)) if flags[i]]

    dist_list = [init_dist for _ in elev_list]
    sector_list = ["good" for _ in elev_list] # HACK don't define sector names for now

    return dist_list, elev_list, azim_list, sector_list


# ---------------- CAMERAS ----------------------


def init_camera(dist, elev, azim, image_size, device):
    R, T = look_at_view_transform(dist, elev, azim)
    image_size = torch.tensor([image_size, image_size]).unsqueeze(0)
    cameras = PerspectiveCameras(R=R, T=T, device=device, image_size=image_size)

    return cameras

def convert_camera_from_pytorch3d_to_colmap(
    p3d_cameras,
    height,
    width,
    device='cuda',
):
    """From a pytorch3d-compatible camera object and its camera matrices R, T, K, and width, height,
    outputs Gaussian Splatting camera parameters.

    Args:
        p3d_cameras (P3DCameras): R matrices should have shape (N, 3, 3),
            T matrices should have shape (N, 3, 1),
            K matrices should have shape (N, 3, 3).
        height (float): _description_
        width (float): _description_
        device (_type_, optional): _description_. Defaults to 'cuda'.
    """

    N = p3d_cameras.R.shape[0]
    if device is None:
        device = p3d_cameras.device

    if type(height) == torch.Tensor:
        height = int(torch.Tensor([[height.item()]]).to(device))
        width = int(torch.Tensor([[width.item()]]).to(device))
    else:
        height = int(height)
        width = int(width)

    # Inverse extrinsics
    R_inv = (p3d_cameras.R * torch.Tensor([-1.0, 1.0, -1]).to(device)).transpose(-1, -2)
    T_inv = (p3d_cameras.T * torch.Tensor([-1.0, 1.0, -1]).to(device)).unsqueeze(-1)
    world2cam_inv = torch.cat([R_inv, T_inv], dim=-1)
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    world2cam_inv = torch.cat([world2cam_inv, line], dim=-2)
    cam2world_inv = world2cam_inv.inverse()
    camera_to_worlds_inv = cam2world_inv[:, :3]
    
    for cam_idx in range(N):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = camera_to_worlds_inv[cam_idx]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3,:3]
        # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]


    return R, T


def convert_camera_from_pytorch3d_to_gs(
    p3d_cameras,
    height,
    width,
    device='cuda',
):
    """From a pytorch3d-compatible camera object and its camera matrices R, T, K, and width, height,
    outputs Gaussian Splatting camera parameters.

    Args:
        p3d_cameras (P3DCameras): R matrices should have shape (N, 3, 3),
            T matrices should have shape (N, 3, 1),
            K matrices should have shape (N, 3, 3).
        height (float): _description_
        width (float): _description_
        device (_type_, optional): _description_. Defaults to 'cuda'.
    """

    N = p3d_cameras.R.shape[0]
    if device is None:
        device = p3d_cameras.device

    if type(height) == torch.Tensor:
        height = int(torch.Tensor([[height.item()]]).to(device))
        width = int(torch.Tensor([[width.item()]]).to(device))
    else:
        height = int(height)
        width = int(width)

    # Inverse extrinsics
    R_inv = (p3d_cameras.R * torch.Tensor([-1.0, 1.0, -1]).to(device)).transpose(-1, -2)
    T_inv = (p3d_cameras.T * torch.Tensor([-1.0, 1.0, -1]).to(device)).unsqueeze(-1)
    world2cam_inv = torch.cat([R_inv, T_inv], dim=-1)
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    world2cam_inv = torch.cat([world2cam_inv, line], dim=-2)
    cam2world_inv = world2cam_inv.inverse()
    camera_to_worlds_inv = cam2world_inv[:, :3]
    
    for cam_idx in range(N):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = camera_to_worlds_inv[cam_idx]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]


    return R, T