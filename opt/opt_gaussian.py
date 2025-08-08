import torch
from types import SimpleNamespace
from gaussian_renderer import gs_render
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
from tqdm import tqdm
import torchvision
from torchvision import transforms
from PIL import Image
from lib.camera_helper import init_camera, convert_camera_from_pytorch3d_to_colmap
import numpy as np
import time
import trimesh
from depthid_render import get_depth_with_id 
import copy
from torch import nn
import cv2
from depthid_render_mask_control_v2 import get_depth_with_id as get_depth_with_id_2
import matplotlib.pyplot as plt

# Import helper functions from projection_helper
from lib.projection_helper import project_gaussians_to_pixels

threhold = 0.01
num = (990 + 1)
edge_num = 5
middle = torch.log(torch.sqrt(torch.tensor(0.000002))).cuda()

def setup_optimization_environment(scene, view_idx):
    """Set up basic optimization environment."""
    views = scene.getTrainCameras()
    view = views[view_idx]
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    return views, view, pipeline, background

def process_edge_detection(mask_tensor, kernel_size=(5, 5), iterations=2):
    """Process edge detection and dilation."""
    mask_np = np.uint8(mask_tensor.clone().detach().squeeze(0).cpu().numpy() * 255)
    edge = cv2.Canny(mask_np, threshold1=100, threshold2=200)
    dilated = cv2.dilate(edge, np.ones(kernel_size, np.uint8), iterations=iterations)
    _, thresholded = cv2.threshold(mask_np, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_and(thresholded, dilated)
    edges = torch.from_numpy(edges.astype(bool)).bool().cuda().unsqueeze(-1)
    return edges

def process_depth_and_pixel_ids_with_mask(pc_pixel, radii, depth_values, image_size, mask_tensor, 
                                          edge_threshold, use_mask_filter=True, get_depth_func=get_depth_with_id):
    """Process depth and pixel IDs with mask filtering."""
    if get_depth_func == get_depth_with_id_2:
        pix_depth, pix_id = get_depth_func(pc_pixel, radii.float(), depth_values, 
                                          mask_tensor.int().squeeze(0).permute(1, 0).contiguous(), 
                                          image_size, image_size)
    else:
        pix_depth, pix_id = get_depth_func(pc_pixel, radii.float(), depth_values, image_size, image_size)
    
    pix_id = pix_id.permute(1, 0, 2)
    pix_depth = pix_depth.permute(1, 0, 2)
    
    # Process edges
    kernel_size = (7, 7)
    iterations = 3
    edges = process_edge_detection(mask_tensor, kernel_size, iterations)
    
    indices = torch.arange(pix_depth.shape[2]).expand_as(pix_depth).cpu()
    
    pix_id = pix_id.cpu()
    pix_depth = pix_depth.cpu()
    
    # Apply depth threshold
    first_elements = pix_depth[:, :, 0].unsqueeze(-1)
    mask_tmp = pix_depth >= (first_elements + threhold)
    pix_depth[mask_tmp] = -1
    pix_id[mask_tmp] = -1
    pix_id[:, :, num:] = -1
    
    # Apply edge filtering
    pix_id[edges.cpu() & (indices >= edge_threshold)] = -1
    
    # Apply mask filtering
    if use_mask_filter:
        pix_id[(~mask_tensor.bool()).cpu().squeeze(0).unsqueeze(-1) & (indices >= 0)] = -1
    
    return pix_id

def compute_update_tensor(gaussians, pc_pixel, radii, depth_values, image_size, mask_tensor, 
                         tensor_shape):
    """Compute update tensor using dual method approach."""
    # Handle tensor shape parameter
    if isinstance(tensor_shape, list):
        tensor_shape = tensor_shape[0]
    
    update_tensor = torch.zeros(tensor_shape, dtype=torch.bool).cuda()
    
    # First method
    first_get_depth_func = get_depth_with_id
    first_edge_threshold = edge_num
    first_use_mask_filter = True
    
    pix_id1 = process_depth_and_pixel_ids_with_mask(
        pc_pixel, radii, depth_values, image_size, mask_tensor, 
        first_edge_threshold, first_use_mask_filter, first_get_depth_func)
    mask_tmp = (pix_id1 != -1)
    valid_ids = pix_id1[mask_tmp]
    update_tensor[valid_ids.long().cuda()] = True
    
    # Second method
    second_get_depth_func = get_depth_with_id_2
    second_edge_threshold = edge_num
    second_use_mask_filter = True
    
    update_tensor1 = torch.zeros(tensor_shape, dtype=torch.bool).cuda()
    pix_id2 = process_depth_and_pixel_ids_with_mask(
        pc_pixel, radii, depth_values, image_size, mask_tensor, 
        second_edge_threshold, second_use_mask_filter, second_get_depth_func)
    
    mask_tmp = (pix_id2 != -1)
    valid_ids = pix_id2[mask_tmp]
    update_tensor1[valid_ids.long().cuda()] = True
    
    return torch.logical_and(update_tensor, update_tensor1)

def run_optimization_loop(opt_gaussian, view, pipeline, gt_image, mask_tensor, udf_network, 
                         opt, scene, iterations=1500, max_steps=1500):
    """Run the optimization loop."""
    opt.iterations = iterations
    opt.position_lr_max_steps = max_steps
    opt_gaussian.training_setup(opt, scene.cameras_extent)
    
    opt_num = 0
    
    for iteration in tqdm(range(opt.iterations)):
        opt_gaussian.update_learning_rate(iteration)
        background = torch.rand(3).cuda()
        
        render_pkg = gs_render(view, opt_gaussian, pipeline, background)
        image, viewspace_point_tensor, visibility_filter, radii, render_mask = (
            render_pkg["render"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['rendered_alpha']
        )
        
        image = image * mask_tensor
        Ll1 = l1_loss(image, gt_image)
        Ll1_mask = l1_loss(mask_tensor.float(), render_mask)
        scale_loss = torch.mean(torch.square(
            torch.clamp(opt_gaussian._scaling, min=1.1*middle, max=0.9*middle) - opt_gaussian._scaling))
        
        udf = udf_network(opt_gaussian._xyz).mean()
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 
                0.0 * Ll1_mask + 5.0 * scale_loss + 100 * udf)
        
        loss.backward()
        
        with torch.no_grad():
            if iteration < 800:
                opt_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration % 100 == 0 and iteration > 0:
                size_threshold = 20
                opt_num += opt_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.005, 
                                                        scene.cameras_extent, size_threshold)
            
            if iteration >= 1000 and opt_gaussian._xyz.grad is not None:
                grad_mask = torch.zeros_like(opt_gaussian._xyz.grad)
                opt_gaussian._xyz.grad *= grad_mask

            if opt_gaussian._xyz.grad is not None:
                grad_mask = torch.ones_like(opt_gaussian._xyz.grad)
                grad_mask[opt_num:, :] = 0
                opt_gaussian._xyz.grad *= grad_mask
            
            if opt_gaussian._opacity.grad is not None:
                opt_gaussian._opacity.grad *= 0

            if opt_gaussian._rotation.grad is not None:
                opt_gaussian._rotation.grad *= 0
                
            opt_gaussian.optimizer.step()
            opt_gaussian.optimizer.zero_grad(set_to_none=True)
    

def merge_gaussian_parameters(gaussians, opt_gaussian, update_tensor, new_gaussian, view_idx, is_initial=False):
    """Merge optimized gaussian parameters back to original gaussians."""
    with torch.no_grad():
        gaussians.num_no_opt = gaussians._xyz[~update_tensor].shape[0]
        gaussians._xyz = nn.Parameter(torch.cat((gaussians._xyz[~update_tensor], opt_gaussian._xyz), dim=0).requires_grad_(True))
        gaussians._features_dc = nn.Parameter(torch.cat((gaussians._features_dc[~update_tensor], opt_gaussian._features_dc), dim=0).requires_grad_(True))
        gaussians._features_rest = nn.Parameter(torch.cat((gaussians._features_rest[~update_tensor], opt_gaussian._features_rest), dim=0).requires_grad_(True))
        gaussians._opacity = nn.Parameter(torch.cat((gaussians._opacity[~update_tensor], opt_gaussian._opacity), dim=0).requires_grad_(True))
        gaussians._scaling = nn.Parameter(torch.cat((gaussians._scaling[~update_tensor], opt_gaussian._scaling), dim=0).requires_grad_(True))
        gaussians._rotation = nn.Parameter(torch.cat((gaussians._rotation[~update_tensor], opt_gaussian._rotation), dim=0).requires_grad_(True))
        gaussians._update[update_tensor] = 1
        gaussians._update = gaussians._update[~update_tensor.bool()]
        gaussians._update = torch.cat((gaussians._update, torch.ones(opt_gaussian._xyz.shape[0]).cuda()), dim=0)
        
        if is_initial and view_idx == 0 and new_gaussian is None:
            new_gaussian = opt_gaussian
        else:
            new_gaussian._xyz = nn.Parameter(torch.cat((new_gaussian._xyz, opt_gaussian._xyz), dim=0).requires_grad_(True))
            new_gaussian._features_dc = nn.Parameter(torch.cat((new_gaussian._features_dc, opt_gaussian._features_dc), dim=0).requires_grad_(True))
            new_gaussian._features_rest = nn.Parameter(torch.cat((new_gaussian._features_rest, opt_gaussian._features_rest), dim=0).requires_grad_(True))
            new_gaussian._opacity = nn.Parameter(torch.cat((new_gaussian._opacity, opt_gaussian._opacity), dim=0).requires_grad_(True))
            new_gaussian._scaling = nn.Parameter(torch.cat((new_gaussian._scaling, opt_gaussian._scaling), dim=0).requires_grad_(True))
            new_gaussian._rotation = nn.Parameter(torch.cat((new_gaussian._rotation, opt_gaussian._rotation), dim=0).requires_grad_(True))
    
    return new_gaussian


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def opt_gaussian_from_one_view(gaussians, scene, view_idx, generate_image, generate_mask_image, opt, init_image, visibility_filter, dist, elev, azim, DEVICE, udf_network, new_gaussian=None):
    views, view, pipeline, background = setup_optimization_environment(scene, view_idx)
    print("Optimize Gaussian")

    init_image = init_image[:,:,:3].permute(2, 0, 1)
    gt_image = (generate_image.permute(2, 0, 1) / 255.0)    
    gt_image = gt_image * generate_mask_image
    
    camera = init_camera(dist, elev, azim, init_image.shape[1], DEVICE)
    R, T = convert_camera_from_pytorch3d_to_colmap(camera, init_image.shape[1], init_image.shape[1])
    
    # Use helper function for 3D to 2D projection
    pc_pixel, points_c_norm = project_gaussians_to_pixels(gaussians, R, T, init_image.shape[1])

    image_size = (init_image.shape[1], init_image.shape[1])
    
    x_grid = np.arange(init_image.shape[1])
    y_grid = np.arange(init_image.shape[1])
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_x = torch.from_numpy(grid_x).cuda()
    grid_y = torch.from_numpy(grid_y).cuda()

    rendering_results = gs_render(view, gaussians, pipeline, background)
    radii = rendering_results['radii']
    update_tensor = torch.zeros(visibility_filter.shape, dtype=torch.bool).cuda()

    # Use dual method approach to compute update tensor
    update_tensor = compute_update_tensor(gaussians, pc_pixel, radii, points_c_norm[2,:], 
                                        init_image.shape[1], generate_mask_image, 
                                        visibility_filter.shape)
    
    with torch.no_grad():
        opt_gaussian = copy.deepcopy(gaussians)
        opt_gaussian.select_gaussian(update_tensor)

    run_optimization_loop(opt_gaussian, view, pipeline, gt_image, generate_mask_image, udf_network, opt, scene)

    new_gaussian = merge_gaussian_parameters(gaussians, opt_gaussian, update_tensor, new_gaussian, view_idx, is_initial=True)

    return new_gaussian

def opt_gaussian_from_one_view_update(gaussians, scene, view_idx, opt, init_image, keep_mask_tensor, update_mask_tensor, diffused_image_tensor, dist, elev, azim, DEVICE, udf_network, new_gaussian):
    views, view, pipeline, background = setup_optimization_environment(scene, view_idx)
    print("optim gaussian")

    init_image = init_image[:,:,:3].permute(2, 0, 1) * keep_mask_tensor
    mask = update_mask_tensor.float()
    
    gt_image = (diffused_image_tensor.permute(2, 0, 1) / 255.0) * mask

    camera = init_camera(dist, elev, azim, init_image.shape[1], DEVICE)
    R, T = convert_camera_from_pytorch3d_to_colmap(camera, init_image.shape[1], init_image.shape[1])
    
    # Use helper function for 3D to 2D projection
    pc_pixel, points_c_norm = project_gaussians_to_pixels(gaussians, R, T, init_image.shape[1])

    image_size = (init_image.shape[1], init_image.shape[1])
    
    x_grid = np.arange(init_image.shape[1])
    y_grid = np.arange(init_image.shape[1])
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_x = torch.from_numpy(grid_x).cuda()
    grid_y = torch.from_numpy(grid_y).cuda()

    rendering_results = gs_render(view, gaussians, pipeline, background)
    radii = rendering_results['radii']
    update_tensor = torch.zeros([gaussians._xyz.shape[0]], dtype=torch.bool).cuda()

    # Use dual method approach to compute update tensor (reversed logic for update function)
    update_tensor = compute_update_tensor(gaussians, pc_pixel, radii, points_c_norm[2,:], 
                                        init_image.shape[1], mask, 
                                        gaussians._xyz.shape[0])


    with torch.no_grad():
        opt_gaussian = copy.deepcopy(gaussians)
        opt_gaussian.select_gaussian(update_tensor)
        opt_gaussian.reset_color()
    
    run_optimization_loop(opt_gaussian, view, pipeline, gt_image, mask, udf_network, opt, scene)

    new_gaussian = merge_gaussian_parameters(gaussians, opt_gaussian, update_tensor, new_gaussian, view_idx, is_initial=False)
    
    return new_gaussian

