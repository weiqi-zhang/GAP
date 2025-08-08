
import argparse
import os
import sys
import time
from types import SimpleNamespace


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(".")


import numpy as np
import torch
import torchvision
from PIL import Image


from arguments import OptimizationParams


from gaussian_renderer import gs_render
from scene import Scene
from scene.fields import CAPUDFNetwork
from scene.gaussian_model import GaussianModel


from opt.inpainting import update_colored_points
from opt.opt_gaussian import (
    opt_gaussian_from_one_view,
    opt_gaussian_from_one_view_update
)

from lib.camera_helper import (
    init_camera,
    init_viewpoints,
    convert_camera_from_pytorch3d_to_colmap,
    convert_camera_from_pytorch3d_to_gs
)
from lib.diffusion_helper import (
    apply_controlnet_depth,
    get_controlnet_depth
)
from lib.io_helper import save_args
from lib.projection_helper import (
    build_similarity_gaussian_cache_for_all_views_gaussian_udf,
    render_one_view_and_build_masks_gaussian
)
from lib.vis_helper import visualize_principle_viewpoints


from CAP.run import Runner
from init.point2gs_sh3_2dgs import conver_ply
from init.sample_points import sample_points

# 设备设置
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()


def init_args():
    print("=> initializing input arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--pc_name", type=str, required=True)
    parser.add_argument("--pc_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--a_prompt", type=str, default="best quality, high quality, extremely detailed, good geometry, pure color, less strip")
    parser.add_argument("--n_prompt", type=str, default="deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke")
    parser.add_argument("--new_strength", type=float, default=1)
    parser.add_argument("--update_strength", type=float, default=0.5)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=10)
    parser.add_argument("--num_viewpoints", type=int, default=8)
    parser.add_argument("--viewpoint_mode", type=str, default="predefined", choices=["predefined", "hemisphere"])
    parser.add_argument("--update_steps", type=int, default=8)
    parser.add_argument("--update_mode", type=str, default="heuristic", choices=["sequential", "heuristic", "random"])
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_principle", action="store_true", help="operate on multiple objects")
    parser.add_argument("--use_shapenet", action="store_true", help="operate on ShapeNet objects")
    parser.add_argument("--use_objaverse", action="store_true", help="operate on Objaverse objects")


    # device parameters
    parser.add_argument("--device", type=str, choices=["a6000", "2080"], default="a6000")

    # camera parameters NOTE need careful tuning!!!
    parser.add_argument("--dist", type=float, default=1, 
        help="distance to the camera from the object")
    parser.add_argument("--elev", type=float, default=0,
        help="the angle between the vector from the object to the camera and the horizontal plane")
    parser.add_argument("--azim", type=float, default=180,
        help="the angle between the vector from the object to the camera and the vertical plane")

    args = parser.parse_args()
    op = OptimizationParams(parser)
    
    if args.device == "a6000":
        setattr(args, "image_size", 1024)
        setattr(args, "fragment_k", 1)
    else:
        setattr(args, "image_size", 768)
        setattr(args, "fragment_k", 1)

    return args, op

if __name__ == "__main__":
    args, op = init_args()

    cap_args = SimpleNamespace( 
        mcube_resolution = 512,
        dataname = args.pc_file[:-4],
        dir = args.output_dir,
        input_dir = args.input_dir,
        obj_name = args.pc_name,
    )

    runner = Runner(cap_args, os.path.join(project_root, 'CAP/confs/base.conf'))
    if not os.path.exists(os.path.join(args.output_dir, 'checkpoints', 'ckpt_060000.pth')):
        runner.train()
    # save
    output_dir = os.path.join(
        args.output_dir, 
        "{}-{}-{}-{}-{}".format(
            str(args.seed),
            args.viewpoint_mode[0]+str(args.num_viewpoints),
            args.update_mode[0]+str(args.update_steps),
            str(args.new_strength),
            str(args.update_strength),
        ),
    )

    os.makedirs(output_dir, exist_ok=True)
    print("=> OUTPUT_DIR:", output_dir)

    name = args.pc_file[:-4]
    
    sample_points(os.path.join(args.output_dir, 'mesh', '60000_mesh.obj'), output_dir, 250000)
    conver_ply(output_dir)
    
    gaussians = GaussianModel(0)
    gaussians.load_ply(os.path.join(output_dir, 'point_cloud.ply'))

    udf_network = CAPUDFNetwork(
            d_out=1,
            d_in=3,
            d_hidden=256,
            n_layers=8,
            skip_in=[4],
            multires=0,
            bias=0.5,
            scale=1.0,
            geometric_init=True,
            weight_norm=True
        ).to(DEVICE)

    ckpt_path = os.path.join(args.output_dir, 'checkpoints/ckpt_060000.pth')
    udf_network.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda'))["udf_network_fine"])
    udf_network = udf_network.eval()
    
    
    # initialize viewpoints
    # including: principle viewpoints for generation + refinement viewpoints for updating
    principle_directions = None
    (
        dist_list, 
        elev_list, 
        azim_list, 
        sector_list,
        view_punishments,
        length_viewpoints,
    ) = init_viewpoints(args.viewpoint_mode, args.num_viewpoints, args.dist, args.elev, principle_directions, 
                            use_principle=True, 
                            use_shapenet=args.use_shapenet,
                            use_objaverse=args.use_objaverse)
    
    # save args
    save_args(args, output_dir)

    # initialize depth2image model
    controlnet, ddim_sampler = get_controlnet_depth()
    
    # # ------------------- OPERATION ZONE BELOW ------------------------

    # # 1. generate texture with RePaint 
    # # NOTE no update / refinement

    generate_dir = os.path.join(output_dir, "generate")
    os.makedirs(generate_dir, exist_ok=True)

    init_image_dir = os.path.join(generate_dir, "rendering")
    os.makedirs(init_image_dir, exist_ok=True)

    normal_map_dir = os.path.join(generate_dir, "normal")
    os.makedirs(normal_map_dir, exist_ok=True)

    mask_image_dir = os.path.join(generate_dir, "mask")
    os.makedirs(mask_image_dir, exist_ok=True)

    depth_map_dir = os.path.join(generate_dir, "depth")
    os.makedirs(depth_map_dir, exist_ok=True)

    similarity_map_dir = os.path.join(generate_dir, "similarity")
    os.makedirs(similarity_map_dir, exist_ok=True)

    inpainted_image_dir = os.path.join(generate_dir, "inpainted")
    os.makedirs(inpainted_image_dir, exist_ok=True)

    gaussian_dir = os.path.join(generate_dir, "gaussian")
    os.makedirs(gaussian_dir, exist_ok=True)

    # prepare viewpoints and cache
    NUM_PRINCIPLE = length_viewpoints

    pre_dist_list = dist_list[:NUM_PRINCIPLE]
    pre_elev_list = elev_list[:NUM_PRINCIPLE]
    pre_azim_list = azim_list[:NUM_PRINCIPLE]
    pre_sector_list = sector_list[:NUM_PRINCIPLE]
    pre_view_punishments = view_punishments[:NUM_PRINCIPLE]


    gs_rotations = []
    gs_translations = []
    colmap_rotations = []
    colmap_translations = []
    for view_idx in range(NUM_PRINCIPLE):
        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx]
        camera = init_camera(dist, elev, azim, args.image_size, DEVICE)
        R, T = convert_camera_from_pytorch3d_to_gs(camera, args.image_size, args.image_size)
        gs_rotations.append(R)
        gs_translations.append(T.reshape(3))

        R_c, T_c = convert_camera_from_pytorch3d_to_colmap(camera, args.image_size, args.image_size)
        colmap_rotations.append(R_c)
        colmap_translations.append(T_c)

    scene = Scene(gs_rotations, gs_translations, gaussians, image_size=args.image_size)
    
    gaussians.training_setup(op, scene.cameras_extent)

    similarity_view_cache = build_similarity_gaussian_cache_for_all_views_gaussian_udf(
        pre_dist_list, pre_elev_list, pre_azim_list,
        args.image_size, args.fragment_k,
        DEVICE, udf_network
    )

    # start generation
    print("=> start generating texture...")

    new_gaussian= None
    for view_idx in range(NUM_PRINCIPLE):

        print("=> processing view {}...".format(view_idx))

        # sequentially pop the viewpoints
        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx] 
        prompt = " the {} view of {}".format(sector, args.prompt)

        # 1.1. render and build masks
        (
            view_score,
            cameras,
            init_image, normal_map, depth_map, 
            init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
            keep_mask_image, update_mask_image, generate_mask_image, 
            keep_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor, visibility_filter
        ) = render_one_view_and_build_masks_gaussian(dist, elev, azim, 
            view_idx, view_idx, view_punishments, # => actual view idx and the sequence idx 
            similarity_view_cache,
            args.image_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
            DEVICE, 
            scene,
            udf_network, new_gaussian, colmap_rotations, colmap_translations, gaussians,
            save_intermediate=True
        )
   
    #     # 1.2. generate missing region
    #     # NOTE first view still gets the mask for consistent ablations
        print("=> generating image for prompt: {}...".format(prompt))
        
        if view_idx != 0:
            actual_generate_mask_image = Image.fromarray((np.ones_like(np.array(generate_mask_image)) * 255.).astype(np.uint8))
        else:
            actual_generate_mask_image = generate_mask_image

        print("=> generate for view {}".format(view_idx))
        generate_image, generate_image_before, generate_image_after, generate_image_tensor = apply_controlnet_depth(controlnet, ddim_sampler, 
            init_image.convert("RGBA"), prompt, args.new_strength, args.ddim_steps,
            actual_generate_mask_image, keep_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
            args.a_prompt, args.n_prompt, args.guidance_scale, args.seed, args.eta, 1, DEVICE, args.blend)

        generate_image.save(os.path.join(inpainted_image_dir, "{}.png".format(view_idx)))
        generate_image_before.save(os.path.join(inpainted_image_dir, "{}_before.png".format(view_idx)))
        generate_image_after.save(os.path.join(inpainted_image_dir, "{}_after.png".format(view_idx)))
    #     # 1.2.2 back-project and create texture
    #  
        new_gaussian = opt_gaussian_from_one_view(gaussians, scene, view_idx, generate_image_tensor, generate_mask_tensor, op, init_images_tensor.squeeze(0), visibility_filter, dist, elev, azim, DEVICE, udf_network, new_gaussian)
        
        
        gaussians.save_ply(os.path.join(gaussian_dir, "{}_generate.ply".format(view_idx)))
               
        # 1.2.3. re: render 
        # NOTE only the rendered image is needed - masks should be re-used
        (
            view_score,
            cameras,
            init_image, *_,
        ) = render_one_view_and_build_masks_gaussian(dist, elev, azim, 
            view_idx, view_idx, view_punishments, # => actual view idx and the sequence idx 
            similarity_view_cache,
            args.image_size, args.fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir, gaussian_dir,
            DEVICE, 
            scene,
            udf_network, new_gaussian, colmap_rotations, colmap_translations, gaussians,
            save_intermediate=False,
        )

        views = scene.getTrainCameras()
        view = views[view_idx]
        bg_color = [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        render_pkg = gs_render(view, gaussians, pipeline, background)
        image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        torchvision.utils.save_image(image, os.path.join(gaussian_dir, "{}_generate.png".format(view_idx)))
        

        # 1.3. update blurry region
        # only when: 1) use update flag; 2) there are contents to update; 3) there are enough contexts.
        if update_mask_tensor.sum() > 0 and (update_mask_tensor.sum() / (all_mask_tensor.sum())) > 0.05:
            print("=> update {} pixels for view {}".format(update_mask_tensor.sum().int(), view_idx))
            diffused_image, diffused_image_before, diffused_image_after, diffused_image_tensor = apply_controlnet_depth(controlnet, ddim_sampler, 
                init_image.convert("RGBA"), prompt, args.update_strength, args.ddim_steps,
                update_mask_image, keep_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
                args.a_prompt, args.n_prompt, args.guidance_scale, args.seed, args.eta, 1, DEVICE, args.blend)

            diffused_image.save(os.path.join(inpainted_image_dir, "{}_update.png".format(view_idx)))
            diffused_image_before.save(os.path.join(inpainted_image_dir, "{}_update_before.png".format(view_idx)))
            diffused_image_after.save(os.path.join(inpainted_image_dir, "{}_update_after.png".format(view_idx)))
        
            # 1.3.2. back-project and create texture
            # NOTE projection mask = generate mask
           
            
            new_gaussian = opt_gaussian_from_one_view_update(gaussians, scene, view_idx, op, init_images_tensor.squeeze(0), keep_mask_tensor, update_mask_tensor, diffused_image_tensor, dist, elev, azim, DEVICE, udf_network, new_gaussian)

            gaussians.save_ply(os.path.join(gaussian_dir, "{}_update.ply".format(view_idx)))

            views = scene.getTrainCameras()
            view = views[view_idx]
            bg_color = [0,0,0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
            render_pkg = gs_render(view, gaussians, pipeline, background)
            image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            torchvision.utils.save_image(image, os.path.join(gaussian_dir, "{}_update.png".format(view_idx)))
        else:
            pass
    
    print("Start 3D Inpainting")
    gaussians = update_colored_points(gaussians, gaussian_dir)
    gaussians.save_ply(os.path.join(gaussian_dir, "final.ply".format(view_idx)))
        
    # visualize viewpoints
    visualize_principle_viewpoints(output_dir, pre_dist_list, pre_elev_list, pre_azim_list)

