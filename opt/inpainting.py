import torch
from scipy.spatial import cKDTree
import copy
from torch import nn
import numpy as np
from scipy.spatial.transform import Rotation as R
from kornia.geometry import conversions
import torch.nn.functional as F
import os
from typing import Tuple, Optional


# Constants
EPSILON = 1e-8
DEFAULT_NEIGHBORS = 91
NORMAL_COSINE_THRESHOLD_LOW = 0.5
NORMAL_COSINE_THRESHOLD_HIGH = 0.9
NORMAL_WEIGHT_LOW = 1e-8
NORMAL_WEIGHT_HIGH = 2.0


def quaternion_to_normal(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to normal vectors (unit direction vectors).
    
    Args:
        quaternions: Tensor of shape (N, 4) containing quaternions in WXYZ order
        
    Returns:
        Tensor of shape (N, 3) containing normalized direction vectors
    """
    rotation_matrices = conversions.quaternion_to_rotation_matrix(
        quaternions,
        order=conversions.QuaternionCoeffOrder.WXYZ
    )
    # Extract the Z-axis direction vector (third column) from rotation matrices
    direction_vectors = rotation_matrices[:, :, 2]
    # Normalize direction vectors to ensure they are unit vectors
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    return direction_vectors


def compute_normal_similarity_score(normals1: torch.Tensor, normals2: torch.Tensor) -> torch.Tensor:
    """
    Compute similarity scores between normal vectors using cosine similarity.
    
    Args:
        normals1: First set of normal vectors, shape (N, 3)
        normals2: Second set of normal vectors, shape (N, M, 3)
        
    Returns:
        Cosine similarity scores with special weighting, shape (N, M)
    """
    with torch.no_grad():
        cos_sim = F.cosine_similarity(normals1.unsqueeze(1), normals2, dim=2)
        
        # Apply special weighting based on cosine similarity values
        # Low similarity (< 0.5): assign very small weight
        cos_sim = torch.where(
            (cos_sim >= -1) & (cos_sim < NORMAL_COSINE_THRESHOLD_LOW), 
            torch.tensor(NORMAL_WEIGHT_LOW, device=cos_sim.device), 
            cos_sim
        )
        
        # High similarity (>= 0.9): assign high weight
        cos_sim = torch.where(
            (cos_sim >= NORMAL_COSINE_THRESHOLD_HIGH) & (cos_sim <= 1), 
            torch.tensor(NORMAL_WEIGHT_HIGH, device=cos_sim.device), 
            cos_sim
        )

    return cos_sim
    

def compute_distance_weights(distances: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized weights based on inverse distances.
    
    Args:
        distances: Distance tensor of shape (N, M)
        
    Returns:
        Normalized weights of shape (N, M)
    """
    # Compute inverse distances with epsilon to avoid division by zero
    inverse_distances = 1 / (distances + EPSILON)
    
    # Normalize weights so they sum to 1 for each point
    sum_inverse_distances = inverse_distances.sum(dim=1, keepdim=True)
    weights = inverse_distances / sum_inverse_distances
    
    return weights


def save_gaussian_subsets(gaussians, gaussian_dir: str, update_tensor: torch.Tensor) -> Tuple:
    """
    Save optimized and non-optimized gaussian subsets to separate files.
    
    Args:
        gaussians: Gaussian model object
        gaussian_dir: Directory to save files
        update_tensor: Boolean mask indicating which gaussians to optimize
        
    Returns:
        Tuple of (no_opt_gaussian, opt_gaussian) objects
    """
    # Create directory if it doesn't exist
    os.makedirs(gaussian_dir, exist_ok=True)
    
    # Create non-optimized gaussian subset
    no_opt_gaussian = copy.deepcopy(gaussians)
    no_opt_gaussian.select_gaussian(~update_tensor.bool())

    # Create optimized gaussian subset
    opt_gaussian = copy.deepcopy(gaussians)
    opt_gaussian.select_gaussian(update_tensor.bool())
    
    return no_opt_gaussian, opt_gaussian


def interpolate_gaussian_features(
    no_opt_gaussian, 
    opt_gaussian, 
    k_neighbors: int = DEFAULT_NEIGHBORS
) -> None:
    """
    Interpolate gaussian features based on spatial proximity and normal similarity.
    
    Args:
        no_opt_gaussian: Non-optimized gaussian subset
        opt_gaussian: Optimized gaussian subset
        k_neighbors: Number of nearest neighbors to consider
    """
    # Build KDTree for efficient nearest neighbor search
    ptree = cKDTree(opt_gaussian._xyz.clone().detach().cpu().numpy())

    # Prepare data arrays
    xyz_array = no_opt_gaussian._xyz.clone().detach().cpu().numpy()
    rotation_array = no_opt_gaussian._rotation.clone().detach().cpu().numpy()
    combined_array = np.concatenate([xyz_array, rotation_array], axis=-1)

    # Process in batches to manage memory
    batch_size = len(combined_array)  # Process all at once for now
    splits = np.array_split(combined_array, 1, axis=0)
    
    for split_idx, batch_data in enumerate(splits):
        no_opt_xyz = batch_data[:, :3]
        no_opt_rot = batch_data[:, 3:7]
        
        # Find k nearest neighbors
        distances, indices = ptree.query(no_opt_xyz, k_neighbors)
        
        # Convert to tensors
        distances = torch.from_numpy(distances).cuda()
        indices = torch.from_numpy(indices).cuda()

        # Extract features from neighboring gaussians
        neighbor_xyz = opt_gaussian._xyz[indices]
        neighbor_features_dc = opt_gaussian._features_dc[indices]
        neighbor_features_rest = opt_gaussian._features_rest[indices]
        neighbor_opacity = opt_gaussian._opacity[indices]
        neighbor_scaling = opt_gaussian._scaling[indices]
        neighbor_rotation = opt_gaussian._rotation[indices]

        # Compute distance-based weights
        distance_weights = compute_distance_weights(distances)

        # Compute normal-based weights
        no_opt_normals = quaternion_to_normal(torch.from_numpy(no_opt_rot).cuda())
        batch_size, num_neighbors = neighbor_rotation.shape[0], neighbor_rotation.shape[1]
        
        opt_normals = quaternion_to_normal(neighbor_rotation.reshape(-1, 4))
        opt_normals = opt_normals.reshape(batch_size, num_neighbors, 3)
        
        normal_weights = compute_normal_similarity_score(no_opt_normals, opt_normals)
        
        # Compute opacity-based weights
        opacity_weights = (neighbor_opacity / neighbor_opacity.max()).squeeze(-1)
        
        # Combine all weights
        combined_weights = distance_weights * normal_weights * opacity_weights

        # Interpolate features using weighted averages
        # Update DC features
        weighted_dc = combined_weights.unsqueeze(-1).unsqueeze(-1) * neighbor_features_dc
        no_opt_gaussian._features_dc = weighted_dc.sum(dim=1).float()
        
        # Update rest features
        weighted_rest = combined_weights.unsqueeze(-1).unsqueeze(-1) * neighbor_features_rest
        no_opt_gaussian._features_rest = weighted_rest.sum(dim=1).float()


def update_colored_points_rob(gaussians, gaussian_dir: str) -> object:
    """
    Update gaussian points by interpolating colors from optimized regions.
    
    This function separates gaussians into optimized and non-optimized groups,
    then updates the non-optimized gaussians by interpolating features from
    nearby optimized gaussians based on spatial proximity, normal similarity,
    and opacity weights.
    
    Args:
        gaussians: Gaussian model object containing all gaussian points
        gaussian_dir: Directory path to save intermediate gaussian files
        
    Returns:
        Updated gaussians object with interpolated features
    """
    # Get update mask
    update_tensor = gaussians._update.clone().bool()
    
    # Save and create gaussian subsets
    no_opt_gaussian, opt_gaussian = save_gaussian_subsets(
        gaussians, gaussian_dir, update_tensor
    )
    
    # Interpolate features for non-optimized gaussians
    interpolate_gaussian_features(no_opt_gaussian, opt_gaussian)
    
    # Concatenate updated gaussians back together
    gaussians._xyz = nn.Parameter(
        torch.cat((no_opt_gaussian._xyz, opt_gaussian._xyz), dim=0).requires_grad_(True)
    )
    gaussians._features_dc = nn.Parameter(
        torch.cat((no_opt_gaussian._features_dc, opt_gaussian._features_dc), dim=0).requires_grad_(True)
    )
    gaussians._features_rest = nn.Parameter(
        torch.cat((no_opt_gaussian._features_rest, opt_gaussian._features_rest), dim=0).requires_grad_(True)
    )
    gaussians._opacity = nn.Parameter(
        torch.cat((no_opt_gaussian._opacity, opt_gaussian._opacity), dim=0).requires_grad_(True)
    )
    gaussians._scaling = nn.Parameter(
        torch.cat((no_opt_gaussian._scaling, opt_gaussian._scaling), dim=0).requires_grad_(True)
    )
    gaussians._rotation = nn.Parameter(
        torch.cat((no_opt_gaussian._rotation, opt_gaussian._rotation), dim=0).requires_grad_(True)
    )
    
    # Update the optimization mask
    gaussians._update[~update_tensor] = 1
    
    return gaussians

def update_colored_points(gaussians, gaussian_dir):
    update_tensor = gaussians._update.clone().bool()
    no_opt_gaussian = copy.deepcopy(gaussians)
    no_opt_gaussian.select_gaussian(~update_tensor.bool())
    no_opt_gaussian.save_ply(os.path.join(gaussian_dir, 'no_opt_gaussian.ply'))

    opt_gaussian = copy.deepcopy(gaussians)
    opt_gaussian.select_gaussian(update_tensor.bool())
    opt_gaussian.save_ply(os.path.join(gaussian_dir, 'opt_gaussian.ply'))


    ptree = cKDTree(opt_gaussian._xyz.clone().detach().cpu().numpy())
    # sigmas = []

    xyz_array = no_opt_gaussian._xyz.clone().detach().cpu().numpy()  # 转换为 NumPy 数组
    rotation_array = no_opt_gaussian._rotation.clone().detach().cpu().numpy()  # 假设是旋转矩阵或向量
    combined_array = np.concatenate([xyz_array, rotation_array], axis=-1)

    splits = np.array_split(combined_array, 1, axis=0)
    # 遍历分割后的数组
    for split_idx, p in enumerate(splits):
        no_opt_xyz = p[:, :3]
        no_opt_rot = p[:, 3:7]
        distances, indices = ptree.query(no_opt_xyz, 91) # [num_points, N_neighbours]

        distances = torch.from_numpy(distances).cuda()
        indices = torch.from_numpy(indices).cuda()

        features_dc = opt_gaussian._features_dc[indices]
        features_rest = opt_gaussian._features_rest[indices]
        rotation = opt_gaussian._rotation[indices]

        dis_weight = compute_distance_weights(distances)


        no_opt_gaussian_normals = quaternion_to_normal(torch.from_numpy(no_opt_rot).cuda())
        batch, num = rotation.shape[0], rotation.shape[1]
        opt_gaussian_normals = quaternion_to_normal(rotation.reshape(-1, 4))
        opt_gaussian_normals = opt_gaussian_normals.reshape(batch, num, 3)

        normal_weight = compute_normal_similarity_score(no_opt_gaussian_normals, opt_gaussian_normals)

        weight = dis_weight * normal_weight

        # no_opt_gaussian._xyz = weight * xyz
        no_opt_gaussian._features_dc = (weight.unsqueeze(-1).unsqueeze(-1) * features_dc).sum(dim=1).float()
        no_opt_gaussian._features_rest = (weight.unsqueeze(-1).unsqueeze(-1) * features_rest).sum(dim=1).float()
        # no_opt_gaussian._opacity = (weight.unsqueeze(-1).unsqueeze(-1) * opacity).sum(dim=1)
        # no_opt_gaussian._scaling = weight * features_dc
        # no_opt_gaussian._rotation = weight * features_dc

    ptree = cKDTree(gaussians._xyz.clone().detach().cpu().numpy())
    for split_idx, p in enumerate(splits):
        no_opt_xyz = p[:, :3]
        distances, indices = ptree.query(no_opt_xyz, 10) # [num_points, N_neighbours]
        distances = torch.from_numpy(distances).cuda().mean(dim=-1).unsqueeze(-1)
        with torch.no_grad():
            no_opt_gaussian._scaling[:,:] = torch.log(distances)

        neighbors_list = ptree.query_ball_point(no_opt_xyz, r=0.1)
        counts = [len(neighbors) for neighbors in neighbors_list]
        counts_tensor = torch.tensor(counts, dtype=torch.float32).unsqueeze(-1)  # 选择适当的数据类型
        with torch.no_grad():
            no_opt_gaussian._opacity[:,:] = 50000 / counts_tensor

    gaussians._xyz = nn.Parameter(torch.cat((no_opt_gaussian._xyz, opt_gaussian._xyz), dim=0).requires_grad_(True))
    gaussians._features_dc = nn.Parameter(torch.cat((no_opt_gaussian._features_dc, opt_gaussian._features_dc), dim=0).requires_grad_(True))
    gaussians._features_rest = nn.Parameter(torch.cat((no_opt_gaussian._features_rest, opt_gaussian._features_rest), dim=0).requires_grad_(True))
    gaussians._opacity = nn.Parameter(torch.cat((no_opt_gaussian._opacity, opt_gaussian._opacity), dim=0).requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.cat((no_opt_gaussian._scaling, opt_gaussian._scaling), dim=0).requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.cat((no_opt_gaussian._rotation, opt_gaussian._rotation), dim=0).requires_grad_(True))
    gaussians._update[~update_tensor] = 1

    return gaussians
