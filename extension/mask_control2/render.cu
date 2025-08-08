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
    const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits> gs_mask,
    torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits> pix_depth,
    torch::PackedTensorAccessor<int, 3, torch::RestrictPtrTraits>  pix_id
)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    int num=radii.size(0),H=pix_id.size(0),W=pix_id.size(1);
    if(n>=H||m>=W) return ;

    for (int i = 0; i < num; ++i) {
        if (circle_depth[i] < 0||gs_mask[i]==1) continue;//mask过滤

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

__global__ void calcGsMask(const int* mask, int H, int W, const float* pc_pixel, const float* radii, int N, int* gs_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = pc_pixel[idx];
        float y = pc_pixel[idx + N];
        float radius = radii[idx];

        int minX = max(0, static_cast<int>(ceilf(x - radius)));
        int maxX = min(W - 1, static_cast<int>(floorf(x + radius)));
        int minY = max(0, static_cast<int>(ceilf(y - radius)));
        int maxY = min(H - 1, static_cast<int>(floorf(y + radius)));

        for (int i = minY; i <= maxY; ++i) {
            for (int j = minX; j <= maxX; ++j) {
                float dx = j - x;
                float dy = i - y;
                if (dx * dx + dy * dy <= radius * radius&&mask[i*W+j]==0) {
                    gs_mask[idx] = 1;
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
    torch::Tensor gs_mask = torch::zeros({N}, mask.options());//标记每个gs是否被mask过滤
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
    const float* d_circle_pos = circle_pos.data_ptr<float>();
    const float* d_radii = radii.data_ptr<float>();
    const int* d_mask = mask.data_ptr<int>();
    int* d_gs_mask = gs_mask.data_ptr<int>();
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    calcGsMask<<<numBlocks, blockSize>>>(d_mask, H, W, d_circle_pos, d_radii, N, d_gs_mask);
    cudaDeviceSynchronize();
    //stage2 calc id&depth
    const dim3 threads(16, 16);
    const dim3 blocks((H+threads.x-1)/threads.x, (W+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(circle_depth.type(), "get_depth_with_id_fw_cu", 
    ([&] {
        get_depth_with_id_fw_kernel<<<blocks, threads>>>(
                circle_pos.packed_accessor<float, 2, torch::RestrictPtrTraits>(),
                radii.packed_accessor<float, 1, torch::RestrictPtrTraits>(),
                circle_depth.packed_accessor<float, 1, torch::RestrictPtrTraits>(),
                gs_mask.packed_accessor<int, 1, torch::RestrictPtrTraits>(),
                pix_depth.packed_accessor<float, 3, torch::RestrictPtrTraits>(),
                pix_id.packed_accessor<int, 3, torch::RestrictPtrTraits>()
            );
    }));
    cudaDeviceSynchronize();
    return std::make_tuple(pix_depth,pix_id);
}