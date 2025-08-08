#include<torch/extension.h>
#include"utils.h"
std::tuple<torch::Tensor, torch::Tensor> get_depth_with_id(
    torch::Tensor circle_pos,
    torch::Tensor radii,
    torch::Tensor circle_depth,
    int H,int W
){
    CHECK_INPUT(circle_pos);
    CHECK_INPUT(radii);
    CHECK_INPUT(circle_depth);
    std::cout << "circle_pos dtype: " << circle_pos.dtype() << std::endl;
    std::cout << "radii dtype: " << radii.dtype() << std::endl;
    std::cout << "circle_depth dtype: " << circle_depth.dtype() << std::endl;
    return get_depth_with_id_fw_cu(circle_pos,radii,circle_depth,H,W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("get_depth_with_id",&get_depth_with_id);
}