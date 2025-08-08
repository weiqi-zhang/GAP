import torch
import numpy as np
import trimesh

def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = (image_range[1] - image_range[0])/ 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    return pixel_locations, pixel_scaled


def ray_marching(ray0, ray_direction, model, c=None,
                             tau=0.01, n_steps=[349, 350], n_secant_steps=8,
                             depth_range=[0., 2.4], max_points=1000000, rad=1.0):
        ''' Performs ray marching to detect surface points.

        The function returns the surface points as well as d_i of the formula
            ray(d_i) = ray0 + d_i * ray_direction
        which hit the surface points. In addition, masks are returned for
        illegal values.

        Args:
            ray0 (tensor): ray start points of dimension B x N x 3
            ray_direction (tensor):ray direction vectors of dim B x N x 3
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code
            tay (float): threshold value
            n_steps (tuple): interval from which the number of evaluation
                steps if sampled
            n_secant_steps (int): number of secant refinement steps
            depth_range (tuple): range of possible depth values (not relevant when
                using cube intersection)
            method (string): refinement method (default: secant)
            check_cube_intersection (bool): whether to intersect rays with
                unit cube for evaluation
            max_points (int): max number of points loaded to GPU memory
        '''
        # Shotscuts
        batch_size, n_pts, D = ray0.shape
        device = ray0.device
        n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()

            
        depth_intersect, _ = get_sphere_intersection(ray0[:,0], ray_direction, r=rad)
        d_intersect = depth_intersect[...,1]            
        
        d_proposal = torch.linspace(
            0, 1, steps=n_steps).view(
                1, 1, n_steps, 1).to(device)
        d_proposal = depth_range[0] * (1. - d_proposal) + d_intersect.view(1, -1, 1,1)* d_proposal
        

        p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + \
            ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal

        # Evaluate all proposal points in parallel
                
        with torch.no_grad():
            val = torch.cat([(
                model(p_split.squeeze(0)) - tau) for p_split in torch.split(p_proposal.reshape(batch_size, -1, 3),
                    int(max_points / batch_size), dim=1)], dim=0).view(
                        batch_size, -1, n_steps)

        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val[:, :, 0] > 0

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                 torch.ones(batch_size, n_pts, 1).to(device)],
                                dim=-1)

        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0

        mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),
                              torch.arange(n_pts).unsqueeze(-0), indices] > 0
        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied 

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        n = batch_size * n_pts
        d_low = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_high = val.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]

        ray0_masked = ray0[mask]
        ray_direction_masked = ray_direction[mask]

        # write c in pointwise format
        if c is not None and c.shape[-1] != 0:
            c = c.unsqueeze(1).repeat(1, n_pts, 1)[mask]
        
        # Apply surface depth refinement step (e.g. Secant method)
        d_pred = secant(
            f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
            ray_direction_masked, tau, model)
        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts).to(device)
        d_pred_out[mask] = d_pred
        d_pred_out[mask == 0] = np.inf
        d_pred_out[mask_0_not_occupied == 0] = 0
        
        return d_pred_out

def secant(f_low, f_high, d_low, d_high, n_secant_steps,
                        ray0_masked, ray_direction_masked, tau, model, it=0):
    ''' Runs the secant method for interval [d_low, d_high].

    Args:
        d_low (tensor): start values for the interval
        d_high (tensor): end values for the interval
        n_secant_steps (int): number of steps
        ray0_masked (tensor): masked ray start points
        ray_direction_masked (tensor): masked ray direction vectors
        model (nn.Module): model model to evaluate point occupancies
        c (tensor): latent conditioned code c
        tau (float): threshold value in logits
    '''
    d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    for i in range(n_secant_steps):
        p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
        with torch.no_grad():
            f_mid = model(p_mid)[...,0] - tau
        ind_low = f_mid < 0
        ind_low = ind_low
        if ind_low.sum() > 0:
            d_low[ind_low] = d_pred[ind_low]
            f_low[ind_low] = f_mid[ind_low]
        if (ind_low == 0).sum() > 0:
            d_high[ind_low == 0] = d_pred[ind_low == 0]
            f_high[ind_low == 0] = f_mid[ind_low == 0]

        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    return d_pred

def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape
    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0
    
    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect

def gen_rays_at(W, H, pose, intrinsics_inv, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, W - 1, W // l)
        ty = torch.linspace(0, H - 1, H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(intrinsics_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

if __name__ == "__main__":
     
    gen_rays_at(768, 768)
    
