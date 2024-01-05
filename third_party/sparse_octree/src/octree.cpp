#include "../include/octree.h"
#include "../include/utils.h"
#include <queue>
#include <iostream>

// #define MAX_HIT_VOXELS 10
// #define MAX_NUM_VOXELS 10000

int Octant::next_index_ = 0;
// int Octree::feature_index = 0;

int incr_x[8] = {0, 0, 0, 0, 1, 1, 1, 1};
int incr_y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
int incr_z[8] = {0, 1, 0, 1, 0, 1, 0, 1};

Octree::Octree()
{
}

Octree::Octree(int64_t grid_dim, int64_t feat_dim, double voxel_size, std::vector<torch::Tensor> all_pts)
{
    Octant::next_index_ = 0;
    init(grid_dim, feat_dim, voxel_size);
    for (auto &pt : all_pts)
    {
        insert(pt);
    }
}

Octree::~Octree()
{
}

void Octree::init(int64_t grid_dim, int64_t feat_dim, double voxel_size)
{
    size_ = grid_dim;
    feat_dim_ = feat_dim;
    voxel_size_ = voxel_size;
    max_level_ = log2(size_);
    // root_ = std::make_shared<Octant>();
    root_ = new Octant();
    root_->side_ = size_;
    // root_->depth_ = 0;
    root_->is_leaf_ = false;

    // feats_allocated_ = 0;
    // auto options = torch::TensorOptions().requires_grad(true);
    // feats_array_ = torch::randn({MAX_NUM_VOXELS, feat_dim}, options) * 0.01;
}

void Octree::insert(torch::Tensor pts)
{
    // temporal solution
    all_pts.push_back(pts);

    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return;
    }

    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            all_keys.insert(key);

            const unsigned int shift = MAX_BITS - max_level_ - 1;

            auto n = root_;
            unsigned edge = size_ / 2;
            for (int d = 1; d <= max_level_; edge /= 2, ++d)
            {
                const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
                // std::cout << "Level: " << d << " ChildID: " << childid << std::endl;
                auto tmp = n->child(childid);
                if (!tmp)
                {
                    const uint64_t code = key & MASK[d + shift];
                    const bool is_leaf = (d == max_level_);
                    // tmp = std::make_shared<Octant>();
                    tmp = new Octant();
                    tmp->code_ = code;
                    tmp->side_ = edge;
                    tmp->is_leaf_ = is_leaf;
                    tmp->type_ = is_leaf ? (j == 0 ? SURFACE : FEATURE) : NONLEAF;

                    n->children_mask_ = n->children_mask_ | (1 << childid);
                    n->child(childid) = tmp;
                }
                else
                {
                    if (tmp->type_ == FEATURE && j == 0)
                        tmp->type_ = SURFACE;
                }
                n = tmp;
            }
        }
    }
}

double Octree::try_insert(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return -1.0;
    }

    std::set<uint64_t> tmp_keys;

    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            tmp_keys.insert(key);
        }
    }

    std::set<int> result;
    std::set_intersection(all_keys.begin(), all_keys.end(),
                          tmp_keys.begin(), tmp_keys.end(),
                          std::inserter(result, result.end()));

    double overlap_ratio = 1.0 * result.size() / tmp_keys.size();
    return overlap_ratio;
}

Octant *Octree::find_octant(std::vector<float> coord)
{
    int x = int(coord[0]);
    int y = int(coord[1]);
    int z = int(coord[2]);
    // uint64_t key = encode(x, y, z);
    // const unsigned int shift = MAX_BITS - max_level_ - 1;

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return nullptr;

        n = tmp;
    }
    return n;
}

bool Octree::has_voxel(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 1>();
    if (points.size(0) != 3)
    {
        return false;
    }

    int x = int(points[0]);
    int y = int(points[1]);
    int z = int(points[2]);

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return false;

        n = tmp;
    }

    if (!n)
        return false;
    else
        return true;
}

torch::Tensor Octree::get_features(torch::Tensor pts)
{
}

torch::Tensor Octree::get_leaf_voxels()
{
    std::vector<float> voxel_coords = get_leaf_voxel_recursive(root_);

    int N = voxel_coords.size() / 3;
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 3});
    return voxels.clone();
}

std::vector<float> Octree::get_leaf_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    if (n->is_leaf_ && n->type_ == SURFACE)
    {
        auto xyz = decode(n->code_);
        return {xyz[0], xyz[1], xyz[2]};
    }

    std::vector<float> coords;
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_leaf_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

torch::Tensor Octree::get_voxels()
{
    std::vector<float> voxel_coords = get_voxel_recursive(root_);
    int N = voxel_coords.size() / 4;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 4}, options);
    return voxels.clone();
}

std::vector<float> Octree::get_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    auto xyz = decode(n->code_);
    std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(n->side_)};
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

std::pair<int64_t, int64_t> Octree::count_nodes_internal()
{
    return count_recursive_internal(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
std::pair<int64_t, int64_t> Octree::count_recursive_internal(Octant *n)
{
    if (!n)
        return std::make_pair<int64_t, int64_t>(0, 0);

    if (n->is_leaf_)
        return std::make_pair<int64_t, int64_t>(1, 1);

    auto sum = std::make_pair<int64_t, int64_t>(1, 0);

    for (int i = 0; i < 8; i++)
    {
        auto temp = count_recursive_internal(n->child(i));
        sum.first += temp.first;
        sum.second += temp.second;
    }

    return sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Octree::get_centres_and_children()
{
    auto node_count = count_nodes_internal();
    auto total_count = node_count.first;
    auto leaf_count = node_count.second;

    auto all_voxels = torch::zeros({total_count, 4}, dtype(torch::kFloat32));
    auto all_children = -torch::ones({total_count, 8}, dtype(torch::kFloat32));
    auto all_features = -torch::ones({total_count, 8}, dtype(torch::kInt32));

    std::queue<Octant *> all_nodes;
    all_nodes.push(root_);

    while (!all_nodes.empty())
    {
        auto node_ptr = all_nodes.front();
        all_nodes.pop();

        auto xyz = decode(node_ptr->code_);
        std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(node_ptr->side_)};
        auto voxel = torch::from_blob(coords.data(), {4}, dtype(torch::kFloat32));
        all_voxels[node_ptr->index_] = voxel;

        if (node_ptr->type_ == SURFACE)
        {
            for (int i = 0; i < 8; ++i)
            {
                std::vector<float> vcoords = coords;
                vcoords[0] += incr_x[i];
                vcoords[1] += incr_y[i];
                vcoords[2] += incr_z[i];
                auto voxel = find_octant(vcoords);
                if (voxel)
                    all_features.data_ptr<int>()[node_ptr->index_ * 8 + i] = voxel->index_;
            }
        }

        for (int i = 0; i < 8; i++)
        {
            auto child_ptr = node_ptr->child(i);
            if (child_ptr && child_ptr->type_ != FEATURE)
            {
                all_nodes.push(child_ptr);
                all_children[node_ptr->index_][i] = float(child_ptr->index_);
            }
        }
    }

    return std::make_tuple(all_voxels, all_children, all_features);
}

int64_t Octree::count_nodes()
{
    return count_recursive(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
int64_t Octree::count_recursive(Octant *n)
{
    if (!n)
        return 0;

    int64_t sum = 1;

    for (int i = 0; i < 8; i++)
    {
        sum += count_recursive(n->child(i));
    }

    return sum;
}

int64_t Octree::count_leaf_nodes()
{
    return leaves_count_recursive(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
int64_t Octree::leaves_count_recursive(Octant *n)
{
    if (!n)
        return 0;

    if (n->type_ == SURFACE)
    {
        return 1;
    }

    int64_t sum = 0;

    for (int i = 0; i < 8; i++)
    {
        sum += leaves_count_recursive(n->child(i));
    }

    return sum;
}
