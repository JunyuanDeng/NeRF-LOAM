#include "mc_data.cuh"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

__device__ static inline float4 query_sdf_raw(uint bx, uint by, uint bz, uint arx, uint ary, uint arz,
                                              const uint max_vec_num,
                                              const IndexerAccessor indexer,
                                              const CubeSDFRGBAccessor cube_sdf,
                                              const CubeSDFAccessor cube_std,
                                              const BackwardMappingAccessor vec_batch_mapping)
{
    if (bx >= indexer.size(0) || by >= indexer.size(1) || bz >= indexer.size(2))
    {
        return make_float4(NAN, NAN, NAN, NAN);
    }
    //    printf("B-Getting: %d %d %d --> %d, %d, %d\n", bx, by, bz, indexer.size(0), indexer.size(1), indexer.size(2));
    long long vec_ind = indexer[bx][by][bz];
    if (vec_ind == -1 || vec_ind >= max_vec_num)
    {
        return make_float4(NAN, NAN, NAN, NAN);
    }
    int batch_ind = vec_batch_mapping[vec_ind];
    if (batch_ind == -1)
    {
        return make_float4(NAN, NAN, NAN, NAN);
    }
    //    printf("Getting: %d %d %d %d --> %d %d\n", batch_ind, arx, ary, arz, cube_sdf.size(0), cube_sdf.size(1));
    return make_float4(cube_sdf[batch_ind][arx][ary][arz][3],
                       cube_sdf[batch_ind][arx][ary][arz][0],
                       cube_sdf[batch_ind][arx][ary][arz][1],
                       cube_sdf[batch_ind][arx][ary][arz][2]);
}

// Use stddev to weight sdf value.
// #define STD_W_SDF

__device__ static inline float4 get_sdf(const uint3 bsize, const uint r, uint3 bpos, uint3 rpos, const uint max_vec_num,
                                        const IndexerAccessor indexer,
                                        const CubeSDFRGBAccessor cube_sdf,
                                        const CubeSDFAccessor cube_std,
                                        const BackwardMappingAccessor vec_batch_mapping)
{
    if (bpos.x >= bsize.x)
    {
        bpos.x = bsize.x - 1;
        rpos.x = r - 1;
    }
    if (bpos.y >= bsize.y)
    {
        bpos.y = bsize.y - 1;
        rpos.y = r - 1;
    }
    if (bpos.z >= bsize.z)
    {
        bpos.z = bsize.z - 1;
        rpos.z = r - 1;
    }

    if (rpos.x == r)
    {
        bpos.x += 1;
        rpos.x = 0;
    }

    if (rpos.y == r)
    {
        bpos.y += 1;
        rpos.y = 0;
    }

    if (rpos.z == r)
    {
        bpos.z += 1;
        rpos.z = 0;
    }

    float4 total_sdf = query_sdf_raw(bpos.x, bpos.y, bpos.z, rpos.x, rpos.y, rpos.z, max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);

    // If NAN, will also be handled.
    return total_sdf;
}

__device__ static inline float3 sdf_interp(const float3 p1, const float3 p2,
                                           float valp1, float valp2)
{
    if (fabs(0.0f - valp1) < 1.0e-5f)
        return p1;
    if (fabs(0.0f - valp2) < 1.0e-5f)
        return p2;
    if (fabs(valp1 - valp2) < 1.0e-5f)
        return p1;

    float w2 = (0.0f - valp1) / (valp2 - valp1);
    float w1 = 1 - w2;

    return make_float3(p1.x * w1 + p2.x * w2,
                       p1.y * w1 + p2.y * w2,
                       p1.z * w1 + p2.z * w2);
}

__global__ static void meshing_cube_colour(const IndexerAccessor indexer,
                                           const ValidBlocksAccessor valid_blocks,
                                           const BackwardMappingAccessor vec_batch_mapping,
                                           const CubeSDFRGBAccessor cube_sdf,
                                           const CubeSDFAccessor cube_std,
                                           TrianglesAccessor triangles,
                                           TrianglesAccessor vertex_colours,
                                           TriangleVecIdAccessor triangle_flatten_id,
                                           int *__restrict__ triangles_count,
                                           int max_triangles_count,
                                           const uint max_vec_num,
                                           int nx, int ny, int nz,
                                           float max_std)
{
    const uint r = cube_sdf.size(1);
    const uint r3 = r * r * r;
    const uint num_lif = valid_blocks.size(0);
    const float sbs = 1.0f / r; // sub-block-size

    const uint lif_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint sub_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (lif_id >= num_lif || sub_id >= r3)
    {
        return;
    }

    const uint3 bpos = make_uint3(
        (valid_blocks[lif_id] / (ny * nz)) % nx,
        (valid_blocks[lif_id] / nz) % ny,
        valid_blocks[lif_id] % nz);
    const uint3 bsize = make_uint3(indexer.size(0), indexer.size(1), indexer.size(2));
    const uint rx = sub_id / (r * r);
    const uint ry = (sub_id / r) % r;
    const uint rz = sub_id % r;

    // Find all 8 neighbours
    float3 points[8];
    float3 colours[8];
    float sdf_vals[8];

    float4 sdf_val = get_sdf(bsize, r, bpos, make_uint3(rx, ry, rz), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_val.x))
        return;
    sdf_vals[0] = sdf_val.x;
    points[0] = make_float3(bpos.x + rx * sbs, bpos.y + ry * sbs, bpos.z + rz * sbs);
    colours[0] = make_float3(sdf_val.y, sdf_val.z, sdf_val.w);

    sdf_val = get_sdf(bsize, r, bpos, make_uint3(rx + 1, ry, rz), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_val.x))
        return;
    sdf_vals[1] = sdf_val.x;
    points[1] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + ry * sbs, bpos.z + rz * sbs);
    colours[1] = make_float3(sdf_val.y, sdf_val.z, sdf_val.w);

    sdf_val = get_sdf(bsize, r, bpos, make_uint3(rx + 1, ry + 1, rz), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_val.x))
        return;
    sdf_vals[2] = sdf_val.x;
    points[2] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + (ry + 1) * sbs, bpos.z + rz * sbs);
    colours[2] = make_float3(sdf_val.y, sdf_val.z, sdf_val.w);

    sdf_val = get_sdf(bsize, r, bpos, make_uint3(rx, ry + 1, rz), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_val.x))
        return;
    sdf_vals[3] = sdf_val.x;
    points[3] = make_float3(bpos.x + rx * sbs, bpos.y + (ry + 1) * sbs, bpos.z + rz * sbs);
    colours[3] = make_float3(sdf_val.y, sdf_val.z, sdf_val.w);

    sdf_val = get_sdf(bsize, r, bpos, make_uint3(rx, ry, rz + 1), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_val.x))
        return;
    sdf_vals[4] = sdf_val.x;
    points[4] = make_float3(bpos.x + rx * sbs, bpos.y + ry * sbs, bpos.z + (rz + 1) * sbs);
    colours[4] = make_float3(sdf_val.y, sdf_val.z, sdf_val.w);

    sdf_val = get_sdf(bsize, r, bpos, make_uint3(rx + 1, ry, rz + 1), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_val.x))
        return;
    sdf_vals[5] = sdf_val.x;
    points[5] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + ry * sbs, bpos.z + (rz + 1) * sbs);
    colours[5] = make_float3(sdf_val.y, sdf_val.z, sdf_val.w);

    sdf_val = get_sdf(bsize, r, bpos, make_uint3(rx + 1, ry + 1, rz + 1), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_val.x))
        return;
    sdf_vals[6] = sdf_val.x;
    points[6] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + (ry + 1) * sbs, bpos.z + (rz + 1) * sbs);
    colours[6] = make_float3(sdf_val.y, sdf_val.z, sdf_val.w);

    sdf_val = get_sdf(bsize, r, bpos, make_uint3(rx, ry + 1, rz + 1), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_val.x))
        return;
    sdf_vals[7] = sdf_val.x;
    points[7] = make_float3(bpos.x + rx * sbs, bpos.y + (ry + 1) * sbs, bpos.z + (rz + 1) * sbs);
    colours[7] = make_float3(sdf_val.y, sdf_val.z, sdf_val.w);

    // Find triangle config.
    int cube_type = 0;
    if (sdf_vals[0] < 0)
        cube_type |= 1;
    if (sdf_vals[1] < 0)
        cube_type |= 2;
    if (sdf_vals[2] < 0)
        cube_type |= 4;
    if (sdf_vals[3] < 0)
        cube_type |= 8;
    if (sdf_vals[4] < 0)
        cube_type |= 16;
    if (sdf_vals[5] < 0)
        cube_type |= 32;
    if (sdf_vals[6] < 0)
        cube_type |= 64;
    if (sdf_vals[7] < 0)
        cube_type |= 128;

    // Find vertex position on each edge (weighted by sdf value)
    int edge_config = edgeTable[cube_type];
    float3 vert_list[12];
    float3 rgb_list[12];

    if (edge_config == 0)
        return;
    if (edge_config & 1)
    {
        vert_list[0] = sdf_interp(points[0], points[1], sdf_vals[0], sdf_vals[1]);
        rgb_list[0] = sdf_interp(colours[0], colours[1], sdf_vals[0], sdf_vals[1]);
    }
    if (edge_config & 2)
    {
        vert_list[1] = sdf_interp(points[1], points[2], sdf_vals[1], sdf_vals[2]);
        rgb_list[1] = sdf_interp(colours[1], colours[2], sdf_vals[1], sdf_vals[2]);
    }
    if (edge_config & 4)
    {
        vert_list[2] = sdf_interp(points[2], points[3], sdf_vals[2], sdf_vals[3]);
        rgb_list[2] = sdf_interp(colours[2], colours[3], sdf_vals[2], sdf_vals[3]);
    }
    if (edge_config & 8)
    {
        vert_list[3] = sdf_interp(points[3], points[0], sdf_vals[3], sdf_vals[0]);
        rgb_list[3] = sdf_interp(colours[3], colours[0], sdf_vals[3], sdf_vals[0]);
    }
    if (edge_config & 16)
    {
        vert_list[4] = sdf_interp(points[4], points[5], sdf_vals[4], sdf_vals[5]);
        rgb_list[4] = sdf_interp(colours[4], colours[5], sdf_vals[4], sdf_vals[5]);
    }
    if (edge_config & 32)
    {
        vert_list[5] = sdf_interp(points[5], points[6], sdf_vals[5], sdf_vals[6]);
        rgb_list[5] = sdf_interp(colours[5], colours[6], sdf_vals[5], sdf_vals[6]);
    }
    if (edge_config & 64)
    {
        vert_list[6] = sdf_interp(points[6], points[7], sdf_vals[6], sdf_vals[7]);
        rgb_list[6] = sdf_interp(colours[6], colours[7], sdf_vals[6], sdf_vals[7]);
    }
    if (edge_config & 128)
    {
        vert_list[7] = sdf_interp(points[7], points[4], sdf_vals[7], sdf_vals[4]);
        rgb_list[7] = sdf_interp(colours[7], colours[4], sdf_vals[7], sdf_vals[4]);
    }
    if (edge_config & 256)
    {
        vert_list[8] = sdf_interp(points[0], points[4], sdf_vals[0], sdf_vals[4]);
        rgb_list[8] = sdf_interp(colours[0], colours[4], sdf_vals[0], sdf_vals[4]);
    }
    if (edge_config & 512)
    {
        vert_list[9] = sdf_interp(points[1], points[5], sdf_vals[1], sdf_vals[5]);
        rgb_list[9] = sdf_interp(colours[1], colours[5], sdf_vals[1], sdf_vals[5]);
    }
    if (edge_config & 1024)
    {
        vert_list[10] = sdf_interp(points[2], points[6], sdf_vals[2], sdf_vals[6]);
        rgb_list[10] = sdf_interp(colours[2], colours[6], sdf_vals[2], sdf_vals[6]);
    }
    if (edge_config & 2048)
    {
        vert_list[11] = sdf_interp(points[3], points[7], sdf_vals[3], sdf_vals[7]);
        rgb_list[11] = sdf_interp(colours[3], colours[7], sdf_vals[3], sdf_vals[7]);
    }

    // Write triangles to array.
    float3 vp[3];
    float3 vc[3];
    for (int i = 0; triangleTable[cube_type][i] != -1; i += 3)
    {
#pragma unroll
        for (int vi = 0; vi < 3; ++vi)
        {
            vp[vi] = vert_list[triangleTable[cube_type][i + vi]];
            vc[vi] = rgb_list[triangleTable[cube_type][i + vi]];
        }

        int triangle_id = atomicAdd(triangles_count, 1);
        if (triangle_id < max_triangles_count)
        {
#pragma unroll
            for (int vi = 0; vi < 3; ++vi)
            {
                triangles[triangle_id][vi][0] = vp[vi].x;
                triangles[triangle_id][vi][1] = vp[vi].y;
                triangles[triangle_id][vi][2] = vp[vi].z;
                vertex_colours[triangle_id][vi][0] = vc[vi].x;
                vertex_colours[triangle_id][vi][1] = vc[vi].y;
                vertex_colours[triangle_id][vi][2] = vc[vi].z;
            }
            triangle_flatten_id[triangle_id] = valid_blocks[lif_id];
        }
    }
}

std::vector<torch::Tensor> marching_cubes_sparse_colour(
    torch::Tensor indexer,           // (nx, ny, nz) -> data_id
    torch::Tensor valid_blocks,      // (K, )
    torch::Tensor vec_batch_mapping, //
    torch::Tensor cube_rgb_sdf,      // (M, rx, ry, rz, 4)
    torch::Tensor cube_std,          // (M, rx, ry, rz)
    const std::vector<int> &n_xyz,   // [nx, ny, nz]
    float max_std,                   // Prune all vertices
    int max_n_triangles              // Maximum number of triangle buffer
)
{
    CHECK_INPUT(indexer);
    CHECK_INPUT(valid_blocks);
    CHECK_INPUT(cube_rgb_sdf);
    CHECK_INPUT(cube_std);
    CHECK_INPUT(vec_batch_mapping);
    assert(max_n_triangles > 0);

    const int r = cube_rgb_sdf.size(1);
    const int r3 = r * r * r;
    const int num_lif = valid_blocks.size(0);
    const uint max_vec_num = vec_batch_mapping.size(0);

    torch::Tensor triangles = torch::empty({max_n_triangles, 3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor vertex_colours = torch::empty({max_n_triangles, 3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor triangle_flatten_id = torch::empty({max_n_triangles}, torch::dtype(torch::kLong).device(torch::kCUDA));
    torch::Tensor triangle_std = torch::empty({max_n_triangles, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 dimBlock = dim3(16, 16);
    uint xBlocks = (num_lif + dimBlock.x - 1) / dimBlock.x;
    uint yBlocks = (r3 + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid = dim3(xBlocks, yBlocks);

    thrust::device_vector<int> n_output(1, 0);
    meshing_cube_colour<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        indexer.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
        valid_blocks.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        vec_batch_mapping.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        cube_rgb_sdf.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        cube_std.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        triangles.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        vertex_colours.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        triangle_flatten_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        n_output.data().get(), max_n_triangles, max_vec_num,
        n_xyz[0], n_xyz[1], n_xyz[2], max_std);
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

    int output_n_triangles = n_output[0];
    if (output_n_triangles < max_n_triangles)
    {
        // Trim output tensor if it is not full.
        triangles = triangles.index({torch::indexing::Slice(torch::indexing::None, output_n_triangles)});
        vertex_colours = vertex_colours.index({torch::indexing::Slice(torch::indexing::None, output_n_triangles)});
        triangle_flatten_id = triangle_flatten_id.index({torch::indexing::Slice(torch::indexing::None, output_n_triangles)});
    }
    else
    {
        // Otherwise spawn a warning.
        std::cerr << "Warning from marching cube: the max triangle number is too small " << output_n_triangles << " vs " << max_n_triangles << std::endl;
    }

    return {triangles, vertex_colours, triangle_flatten_id};
}
