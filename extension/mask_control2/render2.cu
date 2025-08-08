#include<torch/extension.h>
#define max_num 1000
__device__ bool is_in(uint2 pix_pos,float2 circle_pos,float radius)
{
    float difx=pix_pos.x-circle_pos.x;
    float dify=pix_pos.y-circle_pos.y;
    return difx*difx+dify*dify<=radius*radius;
}
__global__ void get_depth_with_id_fw_kernel(
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits> circle_pos,
    const torch::PackedTensorAccessor<float, 1, torch::RestrictPtrTraits> radii,
    const torch::PackedTensorAccessor<float, 1, torch::RestrictPtrTraits> circle_depth,
    const torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits> mask,
    torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits> pix_depth,
    torch::PackedTensorAccessor<int, 3, torch::RestrictPtrTraits>  pix_id
)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    int num=radii.size(0),H=pix_id.size(0),W=pix_id.size(1);
    if(n>=H||m>=W) return ;

    for (int i = 0; i < num; ++i) {
        int dx=(int)circle_pos[0][i]+0.5;
        int dy=(int)circle_pos[1][i]+0.5;
        dx=max(0,min(dx,H-1));
        dy=max(0,min(dy,W-1));
        if (circle_depth[i] < 0||mask[dx][dy]==0) continue;//mask过滤

        if (is_in(make_uint2(n, m), make_float2(circle_pos[0][i], circle_pos[1][i]), radii[i])) {
            // 查找适合插入当前深度的索引
            for (int j = 0; j < max_num; ++j) {
                if (pix_id[n][m][j] == -1 || circle_depth[i] < pix_depth[n][m][j]) {
                    // 找到插入的位置
                    for (int k = max_num-1; k > j; --k) {
                        pix_depth[n][m][k] = pix_depth[n][m][k - 1];
                        pix_id[n][m][k] = pix_id[n][m][k - 1];
                    }
                    pix_depth[n][m][j] = circle_depth[i];
                    pix_id[n][m][j] = i;
                    break;
                }
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor> get_depth_with_id_fw_cu(
    torch::Tensor circle_pos,
    torch::Tensor radii,
    torch::Tensor circle_depth,
    torch::Tensor mask,
    int H,int W
){
    const int N=circle_pos.size(1);//获取高斯数量
    torch::Tensor pix_depth = torch::full({H, W,max_num},-1, circle_depth.options());
    torch::Tensor pix_id =  torch::full({H,W,max_num},-1, torch::dtype(torch::kInt32).device("cuda"));
    //torch::Tensor gs_mask = torch::zeros({N}, mask.options());//标记每个gs是否被mask过滤
    //stage1 calc gs_mask
    if (!circle_pos.is_contiguous()) {
        circle_pos = circle_pos.contiguous();
    }
    if (!circle_depth.is_contiguous()) {
        circle_depth = circle_depth.contiguous();
    }
    if (!radii.is_contiguous()) {
        radii = radii.contiguous();
    }
    if (!mask.is_contiguous()) {
        mask = mask.contiguous();
    }
    

    const dim3 threads(16, 16);
    const dim3 blocks((H+threads.x-1)/threads.x, (W+threads.y-1)/threads.y);
    AT_DISPATCH_FLOATING_TYPES(circle_depth.type(), "get_depth_with_id_fw_cu", 
    ([&] {
        get_depth_with_id_fw_kernel<<<blocks, threads>>>(
                circle_pos.packed_accessor<float, 2, torch::RestrictPtrTraits>(),
                radii.packed_accessor<float, 1, torch::RestrictPtrTraits>(),
                circle_depth.packed_accessor<float, 1, torch::RestrictPtrTraits>(),
                mask.packed_accessor<int, 2, torch::RestrictPtrTraits>(),
                pix_depth.packed_accessor<float, 3, torch::RestrictPtrTraits>(),
                pix_id.packed_accessor<int, 3, torch::RestrictPtrTraits>()
            );
    }));
    cudaDeviceSynchronize();
    return std::make_tuple(pix_depth,pix_id);
}