import trimesh
import numpy as np
import torch
from utils.sh_utils import RGB2SH
import open3d as o3d
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import os
from kornia.geometry import conversions
from scipy.spatial.transform import Rotation as R

def convert(data, path):
    xyz = data[:, :3]
    normals = np.zeros_like(xyz)
    f_dc = data[:, 3:6]
    f_rest = data[:, 6:51]
    opacities = data[:,51:52]
    scale = data[:,52:54]
    rotation = data[:,54:58]

    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(2):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

    write_path = path
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(write_path)

# def compute_rotation_quaternions(b):
#     """
#     计算将原始方向 (1,0,0) 旋转到指定方向 b 所需的旋转四元数。
    
#     参数：
#     b (numpy.ndarray): 形状为 [batch_size, 3] 的方向向量数组。
    
#     返回：
#     numpy.ndarray: 形状为 [batch_size, 4] 的四元数数组，格式为 [w, x, y, z]。
#     """
#     a = np.array([0, 1, 0])  # 原始方向
#     b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)  # 归一化 b
#     dot_product = b_norm[:, 0]  # 计算 a 和 b 的点积

#     epsilon = 1e-6  # 允许的数值误差
#     quaternions = np.zeros((b.shape[0], 4))  # 初始化四元数数组

#     # 情况一：方向相同，不需要旋转
#     same_direction = dot_product >= 1.0 - epsilon
#     quaternions[same_direction] = np.array([1.0, 0.0, 0.0, 0.0])

#     # 情况二：方向相反，旋转 180 度
#     opposite_direction = dot_product <= -1.0 + epsilon
#     if np.any(opposite_direction):
#         # 选择一个与 a 垂直的任意轴，这里选择 (0, 1, 0)
#         quaternions[opposite_direction] = np.array([0.0, 0.0, 1.0, 0.0])

#     # 情况三：一般情况，计算旋转四元数
#     general_case = ~(same_direction | opposite_direction)
#     s = np.sqrt((1.0 + dot_product[general_case]) * 0.5)
#     n = np.cross(a, b_norm[general_case])
#     n_norm = n / np.linalg.norm(n, axis=1, keepdims=True)
#     v = n_norm * np.sqrt((1.0 - dot_product[general_case]) * 0.5)[:, np.newaxis]
#     quaternions[general_case] = np.hstack((s[:, np.newaxis], v))

#     return quaternions

# def quaternion_multiply(q1, q2):
#     """
#     计算两个四元数的乘积。

#     参数：
#     q1, q2 (numpy.ndarray): 形状为 [batch_size, 4] 的四元数数组，格式为 [w, x, y, z]。

#     返回：
#     numpy.ndarray: 形状为 [batch_size, 4] 的乘积四元数数组。
#     """
#     w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
#     w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
#     return np.stack((w, x, y, z), axis=1)

# def verify_rotation_quaternions(b, quaternions):
#     """
#     验证计算得到的四元数是否能将原始方向 (1,0,0) 旋转到指定方向 b。

#     参数：
#     b (numpy.ndarray): 形状为 [batch_size, 3] 的方向向量数组。
#     quaternions (numpy.ndarray): 形状为 [batch_size, 4] 的四元数数组，格式为 [w, x, y, z]。

#     返回：
#     None
#     """
#     # 将输入向量 b 归一化
#     b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    
#     # 创建原始向量 (1,0,0)，并重复 batch_size 次
#     v = np.tile(np.array([1.0, 0.0, 0.0]), (b.shape[0], 1))
    
#     # 将向量转换为四元数形式，w 分量为 0
#     v_quat = np.hstack((np.zeros((b.shape[0], 1)), v))
    
#     # 计算四元数的共轭
#     q_conj = quaternions.copy()
#     q_conj[:, 1:] = -q_conj[:, 1:]
    
#     # 旋转向量：v' = q * v * q_conj
#     temp = quaternion_multiply(quaternions, v_quat)
#     v_rotated_quat = quaternion_multiply(temp, q_conj)
#     v_rotated = v_rotated_quat[:, 1:]  # 提取向量部分
    
#     # 计算旋转后的向量与目标向量的误差
#     errors = np.linalg.norm(v_rotated - b_norm, axis=1)
    
#     # 输出误差统计
#     print(f"平均误差: {np.mean(errors)}")
#     print(f"最大误差: {np.max(errors)}")
#     print(f"最小误差: {np.min(errors)}")
#     print(f"误差标准差: {np.std(errors)}")

def create_rotation_matrix_from_direction_vector_batch(direction_vectors):
    # Normalize the batch of direction vectors
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    # Create a batch of arbitrary vectors that are not collinear with the direction vectors
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).to(direction_vectors.device).expand(direction_vectors.shape[0], -1).clone()
    is_collinear = torch.all(torch.abs(direction_vectors - v1) < 1e-5, dim=-1)
    v1[is_collinear] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).to(direction_vectors.device)
    
    # Calculate the first orthogonal vectors
    v1 = torch.cross(direction_vectors, v1)
    v1_norm = torch.norm(v1, dim=-1, keepdim=True)
    v1 = torch.where(v1_norm > 1e-8, v1 / v1_norm, v1)

    # Calculate the second orthogonal vectors by taking the cross product
    v2 = torch.cross(direction_vectors, v1)
    v2_norm = torch.norm(v2, dim=-1, keepdim=True)
    v2 = torch.where(v2_norm > 1e-8, v2 / v2_norm, v2)

    # Create the batch of rotation matrices with the direction vectors as the last columns
    rotation_matrices = torch.stack((v1, v2, direction_vectors), dim=-1)
    return rotation_matrices

def quaternion_to_normal(quaternions):
    rotation_matrices = conversions.quaternion_to_rotation_matrix(
        quaternions,
        order=conversions.QuaternionCoeffOrder.WXYZ
    )

    direction_vectors = rotation_matrices[:, :, 2]
    
    # 标准化方向向量，确保它们是单位向量
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    return direction_vectors

def conver_ply(input_dir):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(os.path.join(input_dir, "example.ply"))
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
    # o3d.io.write_point_cloud("output_file.ply", pcd)

    # 获取点云的坐标
    point_cloud = np.asarray(pcd.points)

    # 获取点云的法线
    normals = np.asarray(pcd.normals)

    colors = np.zeros([point_cloud.shape[0], 3])
    colors[:, 0]  = 0.6
    colors[:, 1]  = 0.6
    colors[:, 2]  = 0.6

    gaussian = np.zeros([point_cloud.shape[0], 58])
    gaussian[:, :3] = point_cloud
    fused_color = RGB2SH(np.asarray(colors))

    features = np.zeros((fused_color.shape[0], 3, (3 + 1) ** 2))
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0


    features_dc = features[:,:,0:1].reshape(features.shape[0], -1)
    features_rest = features[:,:,1:].reshape(features.shape[0], -1)
    gaussian[:,3:6] = features_dc
    gaussian[:, 6:51] = features_rest
    gaussian[:,51] = 5

    gaussian[:,52:54] = -5.8 # 2.753644934974715785741109710242
   
    rotations = create_rotation_matrix_from_direction_vector_batch(torch.from_numpy(normals).float())
    quaternions = conversions.rotation_matrix_to_quaternion(rotations,eps=1e-5, order=conversions.QuaternionCoeffOrder.WXYZ)

    gaussian[:,54:58] = quaternions.numpy()

    convert(gaussian, os.path.join(input_dir, 'point_cloud.ply'))
    os.system(f'rm -rf {os.path.join(input_dir, "example.ply")}')

if __name__ == '__main__':
    conver_ply('/data/zhangweiqi/my_Text2Tex/outputs/ambulance_ae7f73580602427b8352cedf8aaec0fa/44-p40-h20-1.0-0.4-0.1')
    