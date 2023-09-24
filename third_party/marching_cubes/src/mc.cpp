#include <torch/extension.h>

std::vector<torch::Tensor> marching_cubes_sparse(
    torch::Tensor indexer,           // (nx, ny, nz) -> data_id
    torch::Tensor valid_blocks,      // (K, )
    torch::Tensor vec_batch_mapping, //
    torch::Tensor cube_sdf,          // (M, rx, ry, rz)
    torch::Tensor cube_std,          // (M, rx, ry, rz)
    const std::vector<int> &n_xyz,   // [nx, ny, nz]
    float max_std,                   // Prune all vertices
    int max_n_triangles              // Maximum number of triangle buffer.
);

std::vector<torch::Tensor> marching_cubes_sparse_colour(
    torch::Tensor indexer,           // (nx, ny, nz) -> data_id
    torch::Tensor valid_blocks,      // (K, )
    torch::Tensor vec_batch_mapping, //
    torch::Tensor cube_sdf,          // (M, rx, ry, rz, 4)
    torch::Tensor cube_colour,       // (M, rx, ry, rz)
    const std::vector<int> &n_xyz,   // [nx, ny, nz]
    float max_std,                   // Prune all vertices
    int max_n_triangles              // Maximum number of triangle buffer.
);

std::vector<torch::Tensor> marching_cubes_sparse_interp_cuda(
    torch::Tensor indexer,           // (nx, ny, nz) -> data_id
    torch::Tensor valid_blocks,      // (K, )
    torch::Tensor vec_batch_mapping, //
    torch::Tensor cube_sdf,          // (M, rx, ry, rz)
    torch::Tensor cube_std,          // (M, rx, ry, rz)
    const std::vector<int> &n_xyz,   // [nx, ny, nz]
    float max_std,                   // Prune all vertices
    int max_n_triangles              // Maximum number of triangle buffer.
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("marching_cubes_sparse", &marching_cubes_sparse, "Marching Cubes without Interpolation (CUDA)");
    m.def("marching_cubes_sparse_colour", &marching_cubes_sparse_colour, "Marching Cubes without Interpolation (CUDA)");
    m.def("marching_cubes_sparse_interp", &marching_cubes_sparse_interp_cuda, "Marching Cubes with Interpolation (CUDA)");
}