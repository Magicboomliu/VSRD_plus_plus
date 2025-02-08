import torch
import numpy as np
import skimage.measure
import trimesh
from tqdm import tqdm

def generate_mesh_from_sdf_network(sdf_network, grid_size=900, level=0.0, 
                                   bounds=(-60, 80), chunk_size=50000, device="cuda:0"):
    """
    直接在 [-100,100] 3D 网格范围内生成 Mesh (分块计算)
    - sdf_network: 训练好的 SDF 网络，输入 (x, y, z)，输出 SDF 值
    - grid_size: 体素网格大小 (grid_size=200 表示 200x200x200)
    - level: Marching Cubes 提取的 SDF 等值面 (默认 0)
    - bounds: 3D 网格范围
    - chunk_size: 每次送入 SDF 网络的最大点数 (默认 50000)
    - device: 运行 SDF 计算的设备 (CPU / GPU)

    返回:
    - verts: 顶点坐标
    - faces: 三角面片索引
    """

    # 1️⃣ **构建 3D 网格** (在 [-100, 100] 范围内)
    x = torch.linspace(bounds[0], bounds[1], grid_size, device=device)
    y = torch.linspace(bounds[0], bounds[1], grid_size, device=device)
    z = torch.linspace(bounds[0], bounds[1], grid_size, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 2️⃣ **转换为 (N, 3) 形状**
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

    # 3️⃣ **分块计算 SDF**
    sdf_values = torch.zeros(grid_points.shape[0], device=device)  # 预分配内存
    with torch.no_grad():
        for i in tqdm(range(0, grid_points.shape[0], chunk_size)):
            chunk = grid_points[i : i + chunk_size]  # 取出一块数据
            sdf_values_chunk, _ = sdf_network(chunk)
            sdf_values_chunk = sdf_values_chunk.squeeze() 
            sdf_values[i : i + chunk_size] = sdf_values_chunk  # 计算 SDF

    sdf_values = sdf_values.view(grid_size, grid_size, grid_size).cpu().numpy()  # 转 NumPy

    # 4️⃣ **Marching Cubes 提取 Mesh**
    verts, faces, normals, _ = skimage.measure.marching_cubes(
        sdf_values, level=level, 
        spacing=((bounds[1]-bounds[0])/grid_size, 
                 (bounds[1]-bounds[0])/grid_size, 
                 (bounds[1]-bounds[0])/grid_size)
    )

    return verts, faces
