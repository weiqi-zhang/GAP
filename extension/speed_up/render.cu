#include <torch/extension.h>
__device__ bool is_in(uint2 pix_pos, float2 circle_pos, float radius)
{
    float difx = pix_pos.x - circle_pos.x;
    float dify = pix_pos.y - circle_pos.y;
    return difx * difx + dify * dify <= radius * radius;
}
__global__ void get_depth_with_id_fw_kernel(
const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits> circle_pos,
const torch::PackedTensorAccessor<float, 1, torch::RestrictPtrTraits> radii,
const torch::PackedTensorAccessor<float, 1, torch::RestrictPtrTraits> circle_depth,
torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits> pix_depth,
torch::PackedTensorAccessor<int, 3, torch::RestrictPtrTraits> pix_id)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    int num = radii.size(0), H = pix_id.size(0), W = pix_id.size(1);
    if (n >= H || m >= W) return;

    for (int i = 0; i < num; ++i)
    {
        if (circle_depth[i] < 0) continue;

        if (is_in(make_uint2(n, m), make_float2(circle_pos[0][i], circle_pos[1][i]), radii[i]))
        {
         for (int j = 0; j < 1000; ++j)
            {
                if (pix_id[n][m][j] == -1 || circle_depth[i] < pix_depth[n][m][j])
                {
                 for (int k = 999; k > j; --k)
                    {
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
    int H, int W)
{
    const int N = circle_pos.size(1); // 获取高斯数量
    torch::Tensor pix_depth = torch::full({H, W, 1000}, -1, circle_depth.options());
    torch::Tensor pix_id = torch::full({H, W, 1000}, -1, torch::dtype(torch::kInt32).device("cuda"));
    const dim3 threads(16, 16);
    const dim3 blocks((H + threads.x - 1) / threads.x, (W + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(circle_depth.type(), "get_depth_with_id_fw_cu",
                               ([&]
                                { get_depth_with_id_fw_kernel<<<blocks, threads>>>(
                                      circle_pos.packed_accessor<float, 2, torch::RestrictPtrTraits>(),
                                      radii.packed_accessor<float, 1, torch::RestrictPtrTraits>(),
                                      circle_depth.packed_accessor<float, 1, torch::RestrictPtrTraits>(),
                                      pix_depth.packed_accessor<float, 3, torch::RestrictPtrTraits>(),
                                      pix_id.packed_accessor<int, 3, torch::RestrictPtrTraits>()); }));
    cudaDeviceSynchronize();

    // for(int i=0;i<H;++i)
    // {
    //     for(int j=0;j<W;++j)
    //     {
    //         if(pix_depth[i][j][0].item<float>()!=-1)
    //         {
    //             for(int k=0;k<100;++k)
    //             {
    //                 if(pix_depth[i][j][k].item<float>() > (pix_depth[i][j][0].item<float>() +0.1))
    //                 {
    //                     pix_depth[i][j][k]=-1;
    //                     pix_id[i][j][k]=-1;
    //                 }
    //             }
    //         }
    //     }
    // }
    return std::make_tuple(pix_depth, pix_id);
}