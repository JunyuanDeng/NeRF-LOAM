#include "../include/octree.h"
#include "../include/test.h"

TORCH_LIBRARY(svo, m)
{
    m.def("encode", &encode_torch);

    m.class_<Octant>("Octant")
        .def(torch::init<>());

    m.class_<Octree>("Octree")
        .def(torch::init<>())
        .def("init", &Octree::init)
        .def("insert", &Octree::insert)
        .def("try_insert", &Octree::try_insert)
        .def("get_voxels", &Octree::get_voxels)
        .def("get_leaf_voxels", &Octree::get_leaf_voxels)
        .def("get_features", &Octree::get_features)
        .def("count_nodes", &Octree::count_nodes)
        .def("count_leaf_nodes", &Octree::count_leaf_nodes)
        .def("has_voxel", &Octree::has_voxel)
        .def("get_centres_and_children", &Octree::get_centres_and_children)
        .def_pickle(
        // __getstate__
        [](const c10::intrusive_ptr<Octree>& self) -> std::tuple<int64_t, int64_t, double, std::vector<torch::Tensor>> {
            return std::make_tuple(self->size_, self->feat_dim_, self->voxel_size_, self->all_pts);
        },
        // __setstate__
        [](std::tuple<int64_t, int64_t, double, std::vector<torch::Tensor>> state) { 
            return c10::make_intrusive<Octree>(std::get<0>(state), std::get<1>(state), std::get<2>(state), std::get<3>(state));
        });
}