# common utils
import sys
sys.path.append('/data/zhangweiqi/my_Text2tex_fixed')
import os
import argparse
import time
# import matplotlib.pyplot as plt
import math
# pytorch3d
from pytorch3d.renderer import TexturesUV
from tqdm import tqdm
from PIL import Image
# torch
import torch
import trimesh
from torchvision import transforms
from types import SimpleNamespace
# numpy
import numpy as np
# image
from PIL import Image
from arguments import OptimizationParams

from scene import Scene
from gaussian_renderer import gs_render
# customized
import sys
import torchvision
from scene.gaussian_model import (
    GaussianModel,
)

from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
import json

def init_viewpoints():

    # sample space 有多少个需要选择的视角
    # dist_list 距离 写死了
    # elev_list 相机与水平面的夹角
    # azim_list 相机与垂直面的夹角
    with open('/data/zhangweiqi/my_Text2tex_fixed/render/azim.json', 'r') as f:
        loaded_list = json.load(f)
    azim_list = loaded_list
    with open('/data/zhangweiqi/my_Text2tex_fixed/render/elev.json', 'r') as f:
        loaded_list = json.load(f)
    elev_list = loaded_list
    dist_list = [1.1 for i in range(len(elev_list))]
    sector_list = ['#' for i in range(len(elev_list))]
    return  dist_list, elev_list, azim_list, sector_list, len(elev_list)

    

def init_camera(dist, elev, azim, image_size, device):
    R, T = look_at_view_transform(dist, elev, azim)
    image_size = torch.tensor([image_size, image_size]).unsqueeze(0)
    cameras = PerspectiveCameras(R=R, T=T, device=device, image_size=image_size)

    return cameras
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


if __name__ == '__main__':
    lis = os.listdir('/data/genghaotian/my_Text2tex_fixed/dataset_text2tex/outs_p2gs')
    for path in lis:
        
        if os.path.exists(os.path.join('/data/genghaotian/my_Text2tex_fixed/dataset_text2tex/outs_p2gs', path, 'update/gaussian/final.ply')):
            path = os.path.join('/data/genghaotian/my_Text2tex_fixed/dataset_text2tex/outs_p2gs', path, 'update/gaussian/final.ply')
            name = path.split('/')[-4]
            base_dir = '/data/zhangweiqi/my_Text2tex_fixed/render/images'
            output_path = os.path.join(base_dir, name)
            if os.path.exists(output_path):
                continue
            os.makedirs(output_path, exist_ok=True)
            gaussians = GaussianModel(0)
            print(path)
            gaussians.load_ply(path)
            
            (
                dist_list, 
                elev_list, 
                azim_list, 
                sector_list,
                length,
            ) = init_viewpoints()
            
            NUM_PRINCIPLE = length
            pre_dist_list = dist_list[:NUM_PRINCIPLE]
            pre_elev_list = elev_list[:NUM_PRINCIPLE]
            pre_azim_list = azim_list[:NUM_PRINCIPLE]
            pre_sector_list = sector_list[:NUM_PRINCIPLE]
            R_list = []
            T_list = []
            R_raw_list = []
            T_raw_list = []
            for view_idx in range(NUM_PRINCIPLE):
                dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx]
                camera = init_camera(dist, elev, azim, 512, 'cuda')
                R, T = convert_camera_from_pytorch3d_to_gs(camera, 512, 512)
                R_list.append(R)
                T_list.append(T.reshape(3))

            pre_fov = [np.pi / 3.0, np.pi / 4.0, np.pi / 5.0, np.pi / 6.0, np.pi / 7.5]
            for cam_idx in range(0, 5):

                scene = Scene(R_list, T_list, gaussians, image_size=512, fov=pre_fov[cam_idx] * 2)


                for view_idx in range(NUM_PRINCIPLE):
                    views = scene.getTrainCameras()
                    view = views[view_idx]
                    bg_color = [1,1,1]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
                    render_pkg = gs_render(view, gaussians, pipeline, background)
                    image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    torchvision.utils.save_image(image, os.path.join(output_path, f'{view_idx:03d}_{cam_idx}.png'))
            print(f"DDDone!!!")


    