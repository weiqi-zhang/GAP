import torch
import depthid_render
import time
torch.manual_seed(133)
start_time=time.time()
pc_pixel = torch.randn(2, 100).cuda() 

radii = torch.rand(100).cuda() * 10
visibility_filter = torch.ones((800, 800), dtype=torch.bool).cuda()
x_grid = torch.arange(800).cuda()
y_grid = torch.arange(800).cuda()
grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
#print(grid_x.size(),type(grid_x),grid_y)

points_c_norm = torch.randn(4, 100).cuda()

#tmp_time=time.time()
#print(f"代码执行时间为：{tmp_time-start_time}秒")
start2_time=time.time()
print(f"创建变量使用时间为：{start2_time-start_time}秒")
id_tensor=torch.arange(100).cuda()
#按照depth行，给id排序
sort_row = points_c_norm[2, :]
print(sort_row.size())
sorted_indices = torch.argsort(sort_row)
sorted_tensor = id_tensor[sorted_indices]

op_num=0
coords_dict = {}

print(f"sort时间为：{time.time()-start_time}秒")

pix_depth,pix_id=depthid_render.get_depth_with_id(pc_pixel,radii,points_c_norm[2, :],800,800)
print(pix_id.shape)

end_time=time.time()
execution_time = end_time - start_time
print(f"总执行时间为：{execution_time}秒")

mask = pix_id != -1
#y_coords, x_coords = torch.meshgrid(torch.arange(pix_id.size(0)), torch.arange(pix_id.size(1)))
#result = (x_coords[mask] * y_coords[mask]).sum()
# 对结果进行累加
#sum_result = result.sum()
#print(f"coord_sum={result}")
i_indices, j_indices = torch.meshgrid(torch.arange(800, device='cuda'), torch.arange(800, device='cuda'))
valid_mask = pix_id != -1  # 创建掩码，标记有效位置

# 计算 coord_sum
coord_sum = torch.sum((i_indices[valid_mask] * 10 + j_indices[valid_mask]))

print(f"coord_sum={coord_sum}")
print(f"id_sum={(pix_id*mask).sum()}")
print(f"depth_sum={(pix_depth*mask).sum()}")