#include <memory>
#include <torch/script.h>
#include <torch/custom_class.h>

enum OcType
{
    NONLEAF = -1,
    SURFACE = 0,
    FEATURE = 1
};

class Octant : public torch::CustomClassHolder
{
public:
    inline Octant()
    {
        code_ = 0;
        side_ = 0;
        index_ = next_index_++;
        depth_ = -1;
        is_leaf_ = false;
        children_mask_ = 0;
        type_ = NONLEAF;
        for (unsigned int i = 0; i < 8; i++)
        {
            child_ptr_[i] = nullptr;
            // feature_index_[i] = -1;
        }
    }
    ~Octant() {}

    // std::shared_ptr<Octant> &child(const int x, const int y, const int z)
    // {
    //     return child_ptr_[x + y * 2 + z * 4];
    // };

    // std::shared_ptr<Octant> &child(const int offset)
    // {
    //     return child_ptr_[offset];
    // }
    Octant *&child(const int x, const int y, const int z)
    {
        return child_ptr_[x + y * 2 + z * 4];
    };

    Octant *&child(const int offset)
    {
        return child_ptr_[offset];
    }

    uint64_t code_;
    bool is_leaf_;
    unsigned int side_;
    unsigned char children_mask_;
    // std::shared_ptr<Octant> child_ptr_[8];
    // int feature_index_[8];
    int index_;
    int depth_;
    int type_;
    // int feat_index_;
    Octant *child_ptr_[8];
    static int next_index_;
};

class Octree : public torch::CustomClassHolder
{
public:
    Octree();
    // temporal solution
    Octree(int64_t grid_dim, int64_t feat_dim, double voxel_size, std::vector<torch::Tensor> all_pts);
    ~Octree();
    void init(int64_t grid_dim, int64_t feat_dim, double voxel_size);

    // allocate voxels
    void insert(torch::Tensor vox);
    double try_insert(torch::Tensor pts);

    // find a particular octant
    Octant *find_octant(std::vector<float> coord);

    // test intersections
    bool has_voxel(torch::Tensor pose);

    // query features
    torch::Tensor get_features(torch::Tensor pts);

    // get all voxels
    torch::Tensor get_voxels();
    std::vector<float> get_voxel_recursive(Octant *n);

    // get leaf voxels
    torch::Tensor get_leaf_voxels();
    std::vector<float> get_leaf_voxel_recursive(Octant *n);

    // count nodes
    int64_t count_nodes();
    int64_t count_recursive(Octant *n);

    // count leaf nodes
    int64_t count_leaf_nodes();
    // int64_t leaves_count_recursive(std::shared_ptr<Octant> n);
    int64_t leaves_count_recursive(Octant *n);

    // get voxel centres and childrens
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_centres_and_children();

public:
    int size_;
    int feat_dim_;
    int max_level_;

    // temporal solution
    double voxel_size_;
    std::vector<torch::Tensor> all_pts;

private:
    std::set<uint64_t> all_keys;


    // std::shared_ptr<Octant> root_;
    Octant *root_;
    // static int feature_index;

    // internal count function
    std::pair<int64_t, int64_t> count_nodes_internal();
    std::pair<int64_t, int64_t> count_recursive_internal(Octant *n);


};