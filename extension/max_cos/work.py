import torch
import find_max_in_circles
import time
torch.manual_seed(43)
start_time = time.time()
N=100000 #number of gaussian

pc_pixel = torch.rand((2, N), device='cuda')*800
radii = torch.rand(N,device='cuda')* 50

print(pc_pixel)
print(radii)
H, W = 800, 800
cos_similarity = torch.rand((H, W), device='cuda') * 2 - 1#余弦相似度，取值[-1,1]
print(cos_similarity)

results = find_max_in_circles.find_max_in_circles(cos_similarity, pc_pixel, radii)#返回[N,]的tensor

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
print(results[:10],results.dtype)