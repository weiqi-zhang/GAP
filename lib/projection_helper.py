import os
import torch

import cv2
import random

import numpy as np

from torchvision import transforms
from lib.camera_helper import init_camera, convert_camera_from_pytorch3d_to_colmap
import torch.nn.functional as F

from PIL import Image

from tqdm import tqdm
import find_max_in_circles
# customized
import sys
sys.path.append(".")
import copy
from lib.camera_helper import init_camera

from lib.vis_helper import visualize_outputs, visualize_quad_mask
from lib.constants import *
from types import SimpleNamespace
from gaussian_renderer import gs_render
import torchvision
from lib.render import gen_rays_at, ray_marching
from depthid_render import get_depth_with_id

threhold = 0.01
num = (990 + 1)
middle = torch.log(torch.sqrt(torch.tensor(0.000002))).cuda()

def prepare_gaussian_points(gaussians):
    """Prepare 3D points in homogeneous coordinates."""
    points_w = gaussians._xyz.clone()
    ones = torch.ones(gaussians._xyz.shape[0], 1).cuda()
    points_w = torch.cat((points_w, ones), dim=1).permute(1, 0)
    return points_w

def create_world_view_transform(R, T):
    """Create world-to-camera transformation matrix."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    world_view_transform = torch.from_numpy(Rt).cuda().float()
    return world_view_transform

def transform_points_to_camera(points_w, world_view_transform):
    """Transform 3D world points to camera coordinates."""
    points_c = world_view_transform @ points_w
    points_c_norm = points_c / points_c[3:, :]
    return points_c, points_c_norm

def create_camera_intrinsics(image_size):
    """Create camera intrinsics matrix K."""
    K = torch.zeros([3, 4]).cuda()
    focal = image_size / 2
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = image_size / 2
    K[1, 2] = image_size / 2
    K[2, 2] = 1
    return K

def project_to_pixel_coordinates(points_c, K):
    """Project 3D camera points to 2D pixel coordinates."""
    points_pixel = K @ points_c
    points_pixel = points_pixel / points_pixel[2:, :]
    pc_pixel = points_pixel[:2, :]
    return pc_pixel

def setup_rendering_pipeline():
    """Set up the rendering pipeline and background."""
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return pipeline, background

def process_depth_and_pixel_ids(pc_pixel, radii, depth_values, image_size, gaussians):
    """Process depth and pixel IDs to create update tensor."""
    pix_depth, pix_id = get_depth_with_id(pc_pixel, radii.float(), depth_values, image_size, image_size)
    
    pix_id = pix_id.permute(1, 0, 2)
    pix_depth = pix_depth.permute(1, 0, 2)
    
    # Filter by depth threshold
    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = pix_depth >= (first_elements + threhold)
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    pix_id[:, :, num:] = -1
    
    # Create update tensor
    mask_tmp = (pix_id != -1)
    valid_ids = pix_id[mask_tmp]
    update_tensor = torch.zeros([gaussians._xyz.shape[0]], dtype=torch.bool).cuda()
    update_tensor[valid_ids.long()] = True
    update_tensor[~gaussians._update.bool()] = False
    
    return update_tensor

def project_gaussians_to_pixels(gaussians, R, T, image_size):
    """Complete pipeline to project gaussian points to pixel coordinates."""
    points_w = prepare_gaussian_points(gaussians)
    world_view_transform = create_world_view_transform(R, T)
    points_c, points_c_norm = transform_points_to_camera(points_w, world_view_transform)
    K = create_camera_intrinsics(image_size)
    pc_pixel = project_to_pixel_coordinates(points_c, K)
    return pc_pixel, points_c_norm

def setup_raytracing_camera_matrices(R, T, image_size):
    """Set up camera matrices for ray tracing (used in UDF-based functions)."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = T
    pose = np.linalg.inv(pose)
    
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = image_size / 2
    K[1, 1] = image_size / 2
    K[0, 2] = image_size / 2
    K[1, 2] = image_size / 2
    intrinsics_inv = np.linalg.inv(K)
    
    return pose, intrinsics_inv


def get_all_4_locations(values_y, values_x):
    y_0 = torch.floor(values_y)
    y_1 = torch.ceil(values_y)
    x_0 = torch.floor(values_x)
    x_1 = torch.ceil(values_x)

    return torch.cat([y_0, y_0, y_1, y_1], 0).long(), torch.cat([x_0, x_1, x_0, x_1], 0).long()


def compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device):
    """
        compose quad mask:
            -> 0: background
            -> 1: old
            -> 2: update
            -> 3: new
    """

    new_mask_tensor = transforms.ToTensor()(new_mask_image).to(device)
    update_mask_tensor = transforms.ToTensor()(update_mask_image).to(device)
    old_mask_tensor = transforms.ToTensor()(old_mask_image).to(device)

    all_mask_tensor = new_mask_tensor + update_mask_tensor + old_mask_tensor

    quad_mask_tensor = torch.zeros_like(all_mask_tensor)
    quad_mask_tensor[old_mask_tensor == 1] = 1
    quad_mask_tensor[update_mask_tensor == 1] = 2
    quad_mask_tensor[new_mask_tensor == 1] = 3

    return old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor


def compute_view_heat(similarity_tensor, quad_mask_tensor):
    num_total_pixels = quad_mask_tensor.reshape(-1).shape[0]
    heat = 0
    for idx in QUAD_WEIGHTS:
        heat += (quad_mask_tensor == idx).sum() * QUAD_WEIGHTS[idx] / num_total_pixels

    return heat


def select_viewpoint_gaussian(selected_view_ids, view_punishments,
    mode, dist_list, elev_list, azim_list, sector_list, view_idx,
    similarity_texture_cache, exist_texture, similarity_view_cache,
    mesh, faces, verts_uvs,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
    scene, udf_network, new_gaussian, R_raw_list, T_raw_list, gaussians,
    device, use_principle=False
):
    if mode == "sequential":
        
        num_views = len(dist_list)

        dist = dist_list[view_idx % num_views]
        elev = elev_list[view_idx % num_views]
        azim = azim_list[view_idx % num_views]
        sector = sector_list[view_idx % num_views]
        
        selected_view_ids.append(view_idx % num_views)

    elif mode == "heuristic":

        if use_principle and view_idx < 6:

            selected_view_idx = view_idx

        else:

            selected_view_idx = None
            max_heat = 0

            print("=> selecting next view...")
            view_heat_list = []
            for sample_idx in tqdm(range(len(dist_list))):

                view_heat, *_ = render_one_view_and_build_masks_gaussian(dist_list[sample_idx], elev_list[sample_idx], azim_list[sample_idx],
                    sample_idx, sample_idx, view_punishments, # => actual view idx and the sequence idx 
                    similarity_texture_cache, exist_texture, similarity_view_cache,
                    mesh, faces, verts_uvs,
                    image_size, faces_per_pixel,
                    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
                    device, 
                    scene,
                    udf_network, new_gaussian, R_raw_list, T_raw_list, gaussians
                )
                

                if view_heat > max_heat:
                    selected_view_idx = sample_idx
                    max_heat = view_heat

                view_heat_list.append(view_heat.item())

            print(view_heat_list)
            print("select view {} with heat {}".format(selected_view_idx, max_heat))

 
        dist = dist_list[selected_view_idx]
        elev = elev_list[selected_view_idx]
        azim = azim_list[selected_view_idx]
        sector = sector_list[selected_view_idx]

        selected_view_ids.append(selected_view_idx)

        view_punishments[selected_view_idx] *= 0.01

    elif mode == "random":

        selected_view_idx = random.choice(range(len(dist_list)))

        dist = dist_list[selected_view_idx]
        elev = elev_list[selected_view_idx]
        azim = azim_list[selected_view_idx]
        sector = sector_list[selected_view_idx]
        
        selected_view_ids.append(selected_view_idx)

    else:
        raise NotImplementedError()

    return dist, elev, azim, sector, selected_view_ids, view_punishments



@torch.no_grad()
def build_diffusion_mask_gaussian(
    similarity_view_cache, target_value, device, image_size, similarity_map_tensor, visible_mask_tensor, exist_mask_tensor, camera, gaussians, views, R_list, T_list, select_tensor, new_gaussian):
    view = views[target_value]

    # visible mask => the whole region
    visible_mask_tensor = torch.where(visible_mask_tensor > 0.0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    visible_mask_tensor = visible_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
    
    # faces that are too rotated away from the viewpoint will be treated as invisible
    valid_mask_tensor = (similarity_map_tensor >= 0.0).float()

    visible_mask_tensor *= valid_mask_tensor

    # exist mask => visible mask - new mask
    exist_mask_tensor = torch.where(exist_mask_tensor > 0.1, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    exist_mask_tensor = exist_mask_tensor * visible_mask_tensor

    # nonexist mask <=> new mask
    new_mask_tensor = visible_mask_tensor - exist_mask_tensor
    new_mask_tensor[new_mask_tensor < 0] = 0 # NOTE dilate can lead to overflow

    if new_gaussian:
        similarity_idx_cache = torch.zeros([len(R_list), gaussians._xyz.shape[0]]).to(device)
        pipeline, background = setup_rendering_pipeline()
        
        for idx in range(len(R_list)):
            R = R_list[idx]
            T = T_list[idx]
            tmp_view = views[idx]

            # Use the new helper functions
            pc_pixel, points_c_norm = project_gaussians_to_pixels(gaussians, R, T, image_size)
            
            x_grid = np.arange(image_size)
            y_grid = np.arange(image_size)
            grid_x, grid_y = np.meshgrid(x_grid, y_grid)
            grid_x = torch.from_numpy(grid_x).cuda()
            grid_y = torch.from_numpy(grid_y).cuda()

            rendering_results = gs_render(tmp_view, gaussians, pipeline, background)
            radii = rendering_results['radii']

            cos_similarity = similarity_view_cache[idx]
            results = find_max_in_circles.find_max_in_circles(cos_similarity, pc_pixel, radii.float())

            # Use the new helper function for depth and pixel ID processing
            update_tensor = process_depth_and_pixel_ids(pc_pixel, radii, points_c_norm[2,:], image_size, gaussians)
            results[~update_tensor.bool()] = -1
            similarity_idx_cache[idx] = results

        all_update_idx_tensor = (similarity_idx_cache.argmax(0) == target_value).bool()
        sel_gaussian = copy.deepcopy(gaussians)
        all_update_idx_tensor = torch.logical_and(all_update_idx_tensor.bool(), select_tensor.bool())
        sel_gaussian.select_gaussian(all_update_idx_tensor.bool())
        res = gs_render(view, sel_gaussian, pipeline, background)
        all_update_mask_tensor = res['rendered_alpha'].unsqueeze(-1).repeat(1, 1, 1, 3)
        all_update_mask_tensor = torch.where(all_update_mask_tensor > 0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())    
    else:
        all_update_mask_tensor = torch.zeros([1, image_size, image_size, 3]).cuda()
    # all update mask
    
    # print("all_update_mask_tensor ", all_update_mask_tensor.shape)  all_update_mask_tensor  torch.Size([1, 768, 768, 3])

    # current update mask => intersection between all update mask and exist mask
    update_mask_tensor = exist_mask_tensor * all_update_mask_tensor
    # keep mask => exist mask - update mask
    old_mask_tensor = exist_mask_tensor - update_mask_tensor

    # convert
    new_mask = new_mask_tensor[0].cpu().float().permute(2, 0, 1)
    new_mask = transforms.ToPILImage()(new_mask).convert("L")

    update_mask = update_mask_tensor[0].cpu().float().permute(2, 0, 1)
    update_mask = transforms.ToPILImage()(update_mask).convert("L")

    old_mask = old_mask_tensor[0].cpu().float().permute(2, 0, 1)
    old_mask = transforms.ToPILImage()(old_mask).convert("L")

    exist_mask = exist_mask_tensor[0].cpu().float().permute(2, 0, 1)
    exist_mask = transforms.ToPILImage()(exist_mask).convert("L")

    return new_mask, update_mask, old_mask, exist_mask

@torch.no_grad()
def get_camera(
    dist, elev, azim,
    image_size, faces_per_pixel,
    device):

    # render the view
    cameras = init_camera(
        dist, elev, azim,
        image_size, device
    )

    return cameras


def build_similarity_gaussian_cache_for_all_views_gaussian_udf(
    dist_list, elev_list, azim_list,
    image_size,  faces_per_pixel,
    device, udf_network):

    num_candidate_views = len(dist_list)
    similarity_view_cache = torch.zeros(num_candidate_views, image_size, image_size).to(device)

    print("=> building similarity gaussian cache for all views...")
    for i in tqdm(range(num_candidate_views)):
        cameras = get_camera(
            dist_list[i], elev_list[i], azim_list[i],
            image_size, faces_per_pixel, device)
        R, T = convert_camera_from_pytorch3d_to_colmap(cameras, image_size, image_size)
        pose, intrinsics_inv = setup_raytracing_camera_matrices(R, T, image_size)
        rays_o, rays_v = gen_rays_at(image_size, image_size, torch.from_numpy(pose), torch.from_numpy(intrinsics_inv))
        d_pred_out = ray_marching(rays_o.cuda().reshape(1, -1, 3), rays_v.cuda().reshape(1, -1, 3), udf_network, tau=0.01)
        d_pred_out = d_pred_out.reshape(image_size , image_size, 1)
        point = rays_o.cuda() + d_pred_out * rays_v.cuda()
        mask = ~(torch.isnan(point) | torch.isinf(point))
        valid_points = point[mask].reshape(-1, 3)
        gradient = udf_network.gradient(valid_points).squeeze(1)
        norm = torch.norm(gradient, p=2, dim=1, keepdim=True)
        gradient_normalized = gradient / norm

        normal_map = torch.zeros_like(point).cuda()
        normal_map[mask] = gradient_normalized.reshape(-1)
        cosine_similarity = torch.abs(torch.nn.CosineSimilarity(dim=2)(rays_v.cuda(), normal_map.cuda()))

        cosine_similarity = cosine_similarity.detach().cpu().unsqueeze(-1).repeat(1, 1, 3)
        cosine_similarity[~mask] = 0
        similarity_view_cache[i] = cosine_similarity[:,:,0].reshape(image_size, image_size)
        
    return similarity_view_cache

def render_one_view_and_build_masks_gaussian(dist, elev, azim, 
    selected_view_idx, view_idx, view_punishments,
    similarity_view_cache,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
    device, 
    scene,
    udf_network, new_gaussian, R_list, T_list, gaussians,
    save_intermediate=False):
    
    # render the view
    cameras = get_camera(
        dist, elev, azim,
        image_size, faces_per_pixel,
        device
    )

    views = scene.getTrainCameras()
    view = views[selected_view_idx]
    pipeline, background = setup_rendering_pipeline()
    rendering_results = gs_render(view, scene.gaussians, pipeline, background)
    rend_normal = rendering_results['rend_normal']
    torchvision.utils.save_image(rend_normal, os.path.join(normal_map_dir, "{}_rend_normal.png".format(view_idx)))
    surf_normal = rendering_results['surf_normal']
    torchvision.utils.save_image(surf_normal, os.path.join(normal_map_dir, "{}_surf_normal.png".format(view_idx)))
    rend_dist = rendering_results['rend_dist']
    torchvision.utils.save_image(rend_dist, os.path.join(depth_map_dir, "{}_rend_dist.png".format(view_idx)))
    surf_depth = rendering_results['surf_depth']
    torchvision.utils.save_image(surf_depth, os.path.join(depth_map_dir, "{}_surf_depth.png".format(view_idx)))
    torchvision.utils.save_image(rendering_results["render"], os.path.join(init_image_dir, "gs-{}.png".format(view_idx)))
    init_image = Image.open(os.path.join(init_image_dir, "gs-{}.png".format(view_idx))).convert("RGB")
    init_image.save(os.path.join(init_image_dir, "raw-{}.png".format(view_idx)))

    image_array = np.array(init_image)
    init_images_tensor = torch.from_numpy(image_array).unsqueeze(0).cuda() / 255.0
    init_image = Image.fromarray(image_array, 'RGB')

    R, T = convert_camera_from_pytorch3d_to_colmap(cameras, image_size, image_size)
    pose, intrinsics_inv = setup_raytracing_camera_matrices(R, T, image_size)
    rays_o, rays_v = gen_rays_at(image_size, image_size, torch.from_numpy(pose), torch.from_numpy(intrinsics_inv))

    d_pred_out = ray_marching(rays_o.cuda().reshape(1, -1, 3), rays_v.cuda().reshape(1, -1, 3), udf_network, tau=0.01)
    d_pred_out = d_pred_out.reshape(image_size , image_size, 1)
    point = rays_o.cuda() + d_pred_out * rays_v.cuda()
    mask = ~(torch.isnan(point) | torch.isinf(point))
    valid_points = point[mask].reshape(-1, 3)
    gradient = udf_network.gradient(valid_points).squeeze(1)
    norm = torch.norm(gradient, p=2, dim=1, keepdim=True)
    gradient_normalized = gradient / norm
    normal_map = torch.zeros_like(point).cuda()
    normal_map[mask] = gradient_normalized.reshape(-1)
    normal_maps_tensor = normal_map.reshape(image_size, image_size, 3).unsqueeze(0)
    normal_map = normal_maps_tensor.squeeze(0).permute(2, 0, 1)
    cosine_similarity = torch.abs(torch.nn.CosineSimilarity(dim=2)(rays_v.cuda(), normal_map.permute(1, 2, 0).cuda()))
    normal_map = transforms.ToPILImage()(normal_map).convert("RGB")

    similarity_tensor = cosine_similarity.unsqueeze(0).unsqueeze(-1)
    non_zero_similarity = (similarity_tensor > 0).float()
    non_zero_similarity = (non_zero_similarity * 255.).cpu().numpy().astype(np.uint8)[0]
    non_zero_similarity = cv2.erode(non_zero_similarity, kernel=np.ones((3, 3), np.uint8), iterations=2)
    non_zero_similarity = torch.from_numpy(non_zero_similarity).to(similarity_tensor.device).unsqueeze(0) / 255.
    similarity_tensor = non_zero_similarity.unsqueeze(-1) * similarity_tensor

    similarity_map = similarity_tensor[0, :, :, 0].cpu()
    similarity_map = transforms.ToPILImage()(similarity_map).convert("L")

    depth = d_pred_out.reshape(-1)
    depth = torch.where(torch.isnan(depth), torch.tensor(10.0).cuda(), depth)
    depth = torch.where(torch.isinf(depth), torch.tensor(10.0).cuda(), depth)
    no_depth = 8
    pad_value = 0
    depth_min, depth_max = depth[depth < no_depth].min(), depth[depth < no_depth].max()
    target_min, target_max = 15, 255
    depth_value = depth[depth < no_depth]
    depth_value = depth_max - depth_value # reverse values
    depth_value /= (depth_max - depth_min)
    depth_value = depth_value * (target_max - target_min) + target_min
    depth_maps_tensor = depth.clone()
    depth_maps_tensor[depth < no_depth] = depth_value
    depth_maps_tensor[depth >= no_depth] = pad_value
    depth_maps_tensor = depth_maps_tensor.reshape(1, image_size, image_size)
    depth_map = depth_maps_tensor[0].cpu().numpy()
    depth_map = Image.fromarray(depth_map).convert("L")

    radii = rendering_results['radii']
    mid = int((radii.max() + radii.min()) / 2)
    visibility_filter = radii >= mid
    
    rendered_alpha = rendering_results['rendered_alpha'] # torch.Size([1, 1024, 1024]) 
    rendered_alpha = rendered_alpha.reshape(1, image_size, image_size)

    if new_gaussian == None:
        exist_mask_tensor = torch.zeros([1, image_size, image_size, 3]).cuda()
        update_tensor = None
    else:
        # Use the new helper functions
        pc_pixel, points_c_norm = project_gaussians_to_pixels(gaussians, R, T, image_size)
        
        x_grid = np.arange(image_size)
        y_grid = np.arange(image_size)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid_x = torch.from_numpy(grid_x).cuda()
        grid_y = torch.from_numpy(grid_y).cuda()

        rendering_results = gs_render(view, gaussians, pipeline, background)
        radii = rendering_results['radii']
        
        # Use the new helper function for depth and pixel ID processing
        update_tensor = process_depth_and_pixel_ids(pc_pixel, radii, points_c_norm[2,:], image_size, gaussians)

        opt_gaussian = copy.deepcopy(gaussians)
        opt_gaussian.select_gaussian(update_tensor)

        render_res = gs_render(view, opt_gaussian, pipeline, background)
        exist_mask_tensor = render_res['rendered_alpha']
        exist_mask_tensor = exist_mask_tensor.reshape(1, image_size, image_size).unsqueeze(-1).repeat(1, 1, 1, 3)
        exist_mask_tensor[exist_mask_tensor != 0] = 1
        
        #  Hack
        exist_mask_tensor = exist_mask_tensor.permute(0, 3, 1, 2)
        gray = exist_mask_tensor.mean(dim=1, keepdim=True)
        binary = (gray > 0.5).float()
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32).to(exist_mask_tensor.device)
        dilated = F.conv2d(binary, kernel, padding=1)
        eroded = F.conv2d(binary, kernel, padding=1)
        kernel_5 = torch.ones((1, 1, 5, 5), dtype=torch.float32).to(exist_mask_tensor.device)
        eroded_5 = F.conv2d(binary, kernel_5, padding=2)
        eroded_5 = (eroded_5 >= 25).float()
        exist_mask_tensor = eroded_5.permute(0, 2, 3, 1)

    new_mask_image, update_mask_image, old_mask_image, exist_mask_image = build_diffusion_mask_gaussian(
        similarity_view_cache, selected_view_idx, device, image_size, similarity_tensor, depth_maps_tensor, exist_mask_tensor, cameras, gaussians, views, R_list, T_list, update_tensor, new_gaussian)
    # NOTE the view idx is the absolute idx in the sample space (i.e. `selected_view_idx`)
    # it should match with `similarity_texture_cache`

    (
        old_mask_tensor, 
        update_mask_tensor, 
        new_mask_tensor, 
        all_mask_tensor, 
        quad_mask_tensor
    ) = compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device)

    view_heat = compute_view_heat(similarity_tensor, quad_mask_tensor)
    view_heat *= view_punishments[selected_view_idx]

    # save intermediate results
    if save_intermediate:
        init_image.save(os.path.join(init_image_dir, "{}.png".format(view_idx)))
        normal_map.save(os.path.join(normal_map_dir, "{}.png".format(view_idx)))
        depth_map.save(os.path.join(depth_map_dir, "{}.png".format(view_idx)))
        similarity_map.save(os.path.join(similarity_map_dir, "{}.png".format(view_idx)))

        new_mask_image.save(os.path.join(mask_image_dir, "{}_new.png".format(view_idx)))
        update_mask_image.save(os.path.join(mask_image_dir, "{}_update.png".format(view_idx)))
        old_mask_image.save(os.path.join(mask_image_dir, "{}_old.png".format(view_idx)))
        exist_mask_image.save(os.path.join(mask_image_dir, "{}_exist.png".format(view_idx)))

        visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_heat, device)

    return (
        view_heat,
        cameras,
        init_image, normal_map, depth_map, 
        init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
        old_mask_image, update_mask_image, new_mask_image, 
        old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor, visibility_filter
    )
