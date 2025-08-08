#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

__global__ void findMaxInCircles(const float* matrix, int H, int W, const float* pc_pixel, const float* radii, int N, float* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = pc_pixel[idx];
        float y = pc_pixel[idx + N];
        float radius = radii[idx];
        float maxVal = -FLT_MAX;

        int minX = max(0, static_cast<int>(ceilf(x - radius)));
        int maxX = min(W - 1, static_cast<int>(floorf(x + radius)));
        int minY = max(0, static_cast<int>(ceilf(y - radius)));
        int maxY = min(H - 1, static_cast<int>(floorf(y + radius)));

        for (int i = minY; i <= maxY; ++i) {
            for (int j = minX; j <= maxX; ++j) {
                float dx = j - x;
                float dy = i - y;
                if (dx * dx + dy * dy <= radius * radius) {
                    maxVal = fmaxf(maxVal, matrix[i * W + j]);
                }
            }
        }
        results[idx] = maxVal;
    }
}

torch::Tensor find_max_in_circles(torch::Tensor matrix, torch::Tensor pc_pixel, torch::Tensor radii) {
    // 确保Tensor数据是连续的
    if (!matrix.is_contiguous()) {
        matrix = matrix.contiguous();
    }
    if (!pc_pixel.is_contiguous()) {
        pc_pixel = pc_pixel.contiguous();
    }
    if (!radii.is_contiguous()) {
        radii = radii.contiguous();
    }
    int H = matrix.size(0);
    int W = matrix.size(1);
    int N = pc_pixel.size(1);

    //auto results = torch::zeros({N}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    auto results = torch::zeros({N}, matrix.options());

    const float* d_matrix = matrix.data_ptr<float>();
    const float* d_pc_pixel = pc_pixel.data_ptr<float>();
    const float* d_radii = radii.data_ptr<float>();
    float* d_results = results.data_ptr<float>();

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    findMaxInCircles<<<numBlocks, blockSize>>>(d_matrix, H, W, d_pc_pixel, d_radii, N, d_results);

    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("find_max_in_circles", &find_max_in_circles, "Find max in circles (CUDA)");
}