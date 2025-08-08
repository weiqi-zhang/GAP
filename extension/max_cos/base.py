import torch

import time
torch.manual_seed(43)
start_time = time.time()
N=100000 #number of gaussian

pc_pixel = torch.rand((2, N), device='cuda')*800 
radii = torch.rand(N,device='cuda')*50

print(pc_pixel)
print(radii)
H, W = 800, 800
cos_similarity = torch.rand((H, W), device='cuda') * 2 - 1
print(cos_similarity)
result = torch.zeros(N, device='cuda')

for i in range(N):
    x, y = pc_pixel[:, i]
    radius = radii[i]
    
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device='cuda'), torch.arange(W, device='cuda'), indexing='ij')
    mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2
    
    if mask.any():
        result[i] = cos_similarity[mask].max()

#calc time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

print(result[:10],result.dtype)
