import trimesh
import numpy as np
from tqdm import tqdm


def sample_points(in_path, out_path, num_samples):
    # 加载mesh文件，请替换为您的实际文件路径
    mesh = trimesh.load(in_path)  # 支持.obj、.stl等格式

    # 确保mesh是封闭的
    if not mesh.is_watertight:
        mesh = mesh.subdivide()

    # 获取三角形面片和顶点
    faces = mesh.faces
    vertices = mesh.vertices

    # 计算每个三角形的面积
    areas = mesh.area_faces

    # 构建面积的累计分布函数（CDF）
    cdf = np.cumsum(areas)
    cdf /= cdf[-1]

    def sample_mesh_vectorized(mesh, num_samples):
        # 1. 随机选择三角形索引
        r = np.random.rand(num_samples)
        triangle_indices = np.searchsorted(cdf, r)

        # 防止索引超出范围
        triangle_indices = np.clip(triangle_indices, 0, len(faces) - 1)

        # 获取选中的三角形顶点
        selected_faces = faces[triangle_indices]
        v0 = vertices[selected_faces[:, 0]]
        v1 = vertices[selected_faces[:, 1]]
        v2 = vertices[selected_faces[:, 2]]

        # 2. 在每个三角形内部均匀采样点
        u = np.random.rand(num_samples)
        v = np.random.rand(num_samples)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v

        # 计算采样点的坐标
        samples = (v0.T * u + v1.T * v + v2.T * w).T  # Shape: (num_samples, 3)

        # 3. 计算采样点的法向量
        n0 = mesh.vertex_normals[selected_faces[:, 0]]
        n1 = mesh.vertex_normals[selected_faces[:, 1]]
        n2 = mesh.vertex_normals[selected_faces[:, 2]]
        normals = (n0.T * u + n1.T * v + n2.T * w).T
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)  # 避免除以零

        return samples, normals

    # 设置需要采样的点数
    # num_samples = 400000

    # 调用向量化采样函数
    sampled_points, sampled_normals = sample_mesh_vectorized(mesh, num_samples)


    # 保存为PLY文件
    def save_ply(filename, points, normals):
        num_vertices = points.shape[0]
        header = f'''ply
    format ascii 1.0
    element vertex {num_vertices}
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    end_header
    '''
        with open(filename, 'w') as f:
            f.write(header)
            # 使用NumPy的savetxt进行批量写入
            data = np.hstack((points, normals))
            # 使用较高的缓冲区大小提升写入速度
            np.savetxt(f, data, fmt='%.6f %.6f %.6f %.6f %.6f %.6f')

    save_ply(f'{out_path}/example.ply', sampled_points, sampled_normals)

if __name__ == '__main__':
    sample_points('/data/zhangweiqi/gap/data/banana/banana_ada7c35a1a5742f1b4c528eb3daee35b.obj', '/data/zhangweiqi/gap/data/banana', 100000)