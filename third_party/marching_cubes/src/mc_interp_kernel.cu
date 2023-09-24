#include "mc_data.cuh"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

__device__ static inline float2 query_sdf_raw(uint bx, uint by, uint bz, uint arx, uint ary, uint arz,
                                              const uint max_vec_num,
                                              const IndexerAccessor indexer,
                                              const CubeSDFAccessor cube_sdf,
                                              const CubeSDFAccessor cube_std,
                                              const BackwardMappingAccessor vec_batch_mapping)
{
    if (bx >= indexer.size(0) || by >= indexer.size(1) || bz >= indexer.size(2))
    {
        return make_float2(NAN, NAN);
    }
    //    printf("B-Getting: %d %d %d --> %d, %d, %d\n", bx, by, bz, indexer.size(0), indexer.size(1), indexer.size(2));
    long long vec_ind = indexer[bx][by][bz];
    if (vec_ind == -1 || vec_ind >= max_vec_num)
    {
        return make_float2(NAN, NAN);
    }
    int batch_ind = vec_batch_mapping[vec_ind];
    if (batch_ind == -1)
    {
        return make_float2(NAN, NAN);
    }
    //    printf("Getting: %d %d %d %d --> %d %d\n", batch_ind, arx, ary, arz, cube_sdf.size(0), cube_sdf.size(1));
    float sdf = cube_sdf[batch_ind][arx][ary][arz];
    float std = cube_std[batch_ind][arx][ary][arz];
    return make_float2(sdf, std);
}

// Use stddev to weight sdf value.
// #define STD_W_SDF

__device__ static inline float2 get_sdf(const uint3 bsize, const uint r, uint3 bpos, uint3 rpos, const uint max_vec_num,
                                        const IndexerAccessor indexer,
                                        const CubeSDFAccessor cube_sdf,
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

    uint rbound = (r - 1) / 2;
    uint rstart = r / 2;
    float rmid = r / 2.0f;

    float w_xm, w_xp;
    int bxm, rxm, bxp, rxp;
    int zero_x;
    if (rpos.x <= rbound)
    {
        bxm = -1;
        rxm = r;
        bxp = 0;
        rxp = 0;
        w_xp = (float)rpos.x + rmid;
        w_xm = rmid - (float)rpos.x;
        zero_x = 1;
    }
    else
    {
        bxm = 0;
        rxm = 0;
        bxp = 1;
        rxp = -r;
        w_xp = (float)rpos.x - rmid;
        w_xm = rmid + r - (float)rpos.x;
        zero_x = 0;
    }
    w_xm /= r;
    w_xp /= r;

    float w_ym, w_yp;
    int bym, rym, byp, ryp;
    int zero_y;
    if (rpos.y <= rbound)
    {
        bym = -1;
        rym = r;
        byp = 0;
        ryp = 0;
        w_yp = (float)rpos.y + rmid;
        w_ym = rmid - (float)rpos.y;
        zero_y = 1;
    }
    else
    {
        bym = 0;
        rym = 0;
        byp = 1;
        ryp = -r;
        w_yp = (float)rpos.y - rmid;
        w_ym = rmid + r - (float)rpos.y;
        zero_y = 0;
    }
    w_ym /= r;
    w_yp /= r;

    float w_zm, w_zp;
    int bzm, rzm, bzp, rzp;
    int zero_z;
    if (rpos.z <= rbound)
    {
        bzm = -1;
        rzm = r;
        bzp = 0;
        rzp = 0;
        w_zp = (float)rpos.z + rmid;
        w_zm = rmid - (float)rpos.z;
        zero_z = 1;
    }
    else
    {
        bzm = 0;
        rzm = 0;
        bzp = 1;
        rzp = -r;
        w_zp = (float)rpos.z - rmid;
        w_zm = rmid + r - (float)rpos.z;
        zero_z = 0;
    }
    w_zm /= r;
    w_zp /= r;

    rpos.x += rstart;
    rpos.y += rstart;
    rpos.z += rstart;

    // printf("%u %u %u %d %d %d %d %d %d\n", rpos.x, rpos.y, rpos.z, rxm, rxp, rym, ryp, rzm, rzp);

    // Tri-linear interpolation of SDF values.
#ifndef STD_W_SDF
    float total_weight = 0.0;
#else
    float2 total_weight{0.0, 0.0};
#endif
    float2 total_sdf{0.0, 0.0};

    int zero_det = zero_x * 4 + zero_y * 2 + zero_z;

    float2 sdfmmm = query_sdf_raw(bpos.x + bxm, bpos.y + bym, bpos.z + bzm, rpos.x + rxm, rpos.y + rym, rpos.z + rzm,
                                  max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    float wmmm = w_xm * w_ym * w_zm;
#ifndef STD_W_SDF
    if (!isnan(sdfmmm.x))
    {
        total_sdf += sdfmmm * wmmm;
        total_weight += wmmm;
    }
#else
    if (!isnan(sdfmmm.x))
    {
        total_sdf.x += sdfmmm.x * wmmm * sdfmmm.y;
        total_weight.x += wmmm * sdfmmm.y;
        total_sdf.y += wmmm * sdfmmm.y;
        total_weight.y += wmmm;
    }
#endif
    else if (zero_det == 0)
    {
        return make_float2(NAN, NAN);
    }

    float2 sdfmmp = query_sdf_raw(bpos.x + bxm, bpos.y + bym, bpos.z + bzp, rpos.x + rxm, rpos.y + rym, rpos.z + rzp,
                                  max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    float wmmp = w_xm * w_ym * w_zp;
#ifndef STD_W_SDF
    if (!isnan(sdfmmp.x))
    {
        total_sdf += sdfmmp * wmmp;
        total_weight += wmmp;
    }
#else
    if (!isnan(sdfmmp.x))
    {
        total_sdf.x += sdfmmp.x * wmmp * sdfmmp.y;
        total_weight.x += wmmp * sdfmmp.y;
        total_sdf.y += wmmp * sdfmmp.y;
        total_weight.y += wmmp;
    }
#endif
    else if (zero_det == 1)
    {
        return make_float2(NAN, NAN);
    }

    float2 sdfmpm = query_sdf_raw(bpos.x + bxm, bpos.y + byp, bpos.z + bzm, rpos.x + rxm, rpos.y + ryp, rpos.z + rzm,
                                  max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    float wmpm = w_xm * w_yp * w_zm;
#ifndef STD_W_SDF
    if (!isnan(sdfmpm.x))
    {
        total_sdf += sdfmpm * wmpm;
        total_weight += wmpm;
    }
#else
    if (!isnan(sdfmpm.x))
    {
        total_sdf.x += sdfmpm.x * wmpm * sdfmpm.y;
        total_weight.x += wmpm * sdfmpm.y;
        total_sdf.y += wmpm * sdfmpm.y;
        total_weight.y += wmpm;
    }
#endif
    else if (zero_det == 2)
    {
        return make_float2(NAN, NAN);
    }

    float2 sdfmpp = query_sdf_raw(bpos.x + bxm, bpos.y + byp, bpos.z + bzp, rpos.x + rxm, rpos.y + ryp, rpos.z + rzp,
                                  max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    float wmpp = w_xm * w_yp * w_zp;
#ifndef STD_W_SDF
    if (!isnan(sdfmpp.x))
    {
        total_sdf += sdfmpp * wmpp;
        total_weight += wmpp;
    }
#else
    if (!isnan(sdfmpp.x))
    {
        total_sdf.x += sdfmpp.x * wmpp * sdfmpp.y;
        total_weight.x += wmpp * sdfmpp.y;
        total_sdf.y += wmpp * sdfmpp.y;
        total_weight.y += wmpp;
    }
#endif
    else if (zero_det == 3)
    {
        return make_float2(NAN, NAN);
    }

    float2 sdfpmm = query_sdf_raw(bpos.x + bxp, bpos.y + bym, bpos.z + bzm, rpos.x + rxp, rpos.y + rym, rpos.z + rzm,
                                  max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    float wpmm = w_xp * w_ym * w_zm;
#ifndef STD_W_SDF
    if (!isnan(sdfpmm.x))
    {
        total_sdf += sdfpmm * wpmm;
        total_weight += wpmm;
    }
#else
    if (!isnan(sdfpmm.x))
    {
        total_sdf.x += sdfpmm.x * wpmm * sdfpmm.y;
        total_weight.x += wpmm * sdfpmm.y;
        total_sdf.y += wpmm * sdfpmm.y;
        total_weight.y += wpmm;
    }
#endif
    else if (zero_det == 4)
    {
        return make_float2(NAN, NAN);
    }

    float2 sdfpmp = query_sdf_raw(bpos.x + bxp, bpos.y + bym, bpos.z + bzp, rpos.x + rxp, rpos.y + rym, rpos.z + rzp,
                                  max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    float wpmp = w_xp * w_ym * w_zp;
#ifndef STD_W_SDF
    if (!isnan(sdfpmp.x))
    {
        total_sdf += sdfpmp * wpmp;
        total_weight += wpmp;
    }
#else
    if (!isnan(sdfpmp.x))
    {
        total_sdf.x += sdfpmp.x * wpmp * sdfpmp.y;
        total_weight.x += wpmp * sdfpmp.y;
        total_sdf.y += wpmp * sdfpmp.y;
        total_weight.y += wpmp;
    }
#endif
    else if (zero_det == 5)
    {
        return make_float2(NAN, NAN);
    }

    float2 sdfppm = query_sdf_raw(bpos.x + bxp, bpos.y + byp, bpos.z + bzm, rpos.x + rxp, rpos.y + ryp, rpos.z + rzm,
                                  max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    float wppm = w_xp * w_yp * w_zm;
#ifndef STD_W_SDF
    if (!isnan(sdfppm.x))
    {
        total_sdf += sdfppm * wppm;
        total_weight += wppm;
    }
#else
    if (!isnan(sdfppm.x))
    {
        total_sdf.x += sdfppm.x * wppm * sdfppm.y;
        total_weight.x += wppm * sdfppm.y;
        total_sdf.y += wppm * sdfppm.y;
        total_weight.y += wppm;
    }
#endif
    else if (zero_det == 6)
    {
        return make_float2(NAN, NAN);
    }

    float2 sdfppp = query_sdf_raw(bpos.x + bxp, bpos.y + byp, bpos.z + bzp, rpos.x + rxp, rpos.y + ryp, rpos.z + rzp,
                                  max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    float wppp = w_xp * w_yp * w_zp;
#ifndef STD_W_SDF
    if (!isnan(sdfppp.x))
    {
        total_sdf += sdfppp * wppp;
        total_weight += wppp;
    }
#else
    if (!isnan(sdfppp.x))
    {
        total_sdf.x += sdfppp.x * wppp * sdfppp.y;
        total_weight.x += wppp * sdfppp.y;
        total_sdf.y += wppp * sdfppp.y;
        total_weight.y += wppp;
    }
#endif
    else if (zero_det == 7)
    {
        return make_float2(NAN, NAN);
    }

    // If NAN, will also be handled.
    return total_sdf / total_weight;
}

__device__ static inline float4 sdf_interp(const float3 p1, const float3 p2, const float stdp1, const float stdp2,
                                           float valp1, float valp2)
{
    if (fabs(0.0f - valp1) < 1.0e-5f)
        return make_float4(p1, stdp1);
    if (fabs(0.0f - valp2) < 1.0e-5f)
        return make_float4(p2, stdp2);
    if (fabs(valp1 - valp2) < 1.0e-5f)
        return make_float4(p1, stdp1);

    float w2 = (0.0f - valp1) / (valp2 - valp1);
    float w1 = 1 - w2;

    return make_float4(p1.x * w1 + p2.x * w2,
                       p1.y * w1 + p2.y * w2,
                       p1.z * w1 + p2.z * w2,
                       stdp1 * w1 + stdp2 * w2);
}

__global__ static void meshing_cube(const IndexerAccessor indexer,
                                    const ValidBlocksAccessor valid_blocks,
                                    const BackwardMappingAccessor vec_batch_mapping,
                                    const CubeSDFAccessor cube_sdf,
                                    const CubeSDFAccessor cube_std,
                                    TrianglesAccessor triangles,
                                    TriangleStdAccessor triangle_std,
                                    TriangleVecIdAccessor triangle_flatten_id,
                                    int *__restrict__ triangles_count,
                                    int max_triangles_count,
                                    const uint max_vec_num,
                                    int nx, int ny, int nz,
                                    float max_std)
{
    const uint r = cube_sdf.size(1) / 2;
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
    float2 sdf_vals[8];

    sdf_vals[0] = get_sdf(bsize, r, bpos, make_uint3(rx, ry, rz), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_vals[0].x))
        return;
    points[0] = make_float3(bpos.x + rx * sbs, bpos.y + ry * sbs, bpos.z + rz * sbs);

    sdf_vals[1] = get_sdf(bsize, r, bpos, make_uint3(rx + 1, ry, rz), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_vals[1].x))
        return;
    points[1] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + ry * sbs, bpos.z + rz * sbs);

    sdf_vals[2] = get_sdf(bsize, r, bpos, make_uint3(rx + 1, ry + 1, rz), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_vals[2].x))
        return;
    points[2] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + (ry + 1) * sbs, bpos.z + rz * sbs);

    sdf_vals[3] = get_sdf(bsize, r, bpos, make_uint3(rx, ry + 1, rz), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_vals[3].x))
        return;
    points[3] = make_float3(bpos.x + rx * sbs, bpos.y + (ry + 1) * sbs, bpos.z + rz * sbs);

    sdf_vals[4] = get_sdf(bsize, r, bpos, make_uint3(rx, ry, rz + 1), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_vals[4].x))
        return;
    points[4] = make_float3(bpos.x + rx * sbs, bpos.y + ry * sbs, bpos.z + (rz + 1) * sbs);

    sdf_vals[5] = get_sdf(bsize, r, bpos, make_uint3(rx + 1, ry, rz + 1), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_vals[5].x))
        return;
    points[5] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + ry * sbs, bpos.z + (rz + 1) * sbs);

    sdf_vals[6] = get_sdf(bsize, r, bpos, make_uint3(rx + 1, ry + 1, rz + 1), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_vals[6].x))
        return;
    points[6] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + (ry + 1) * sbs, bpos.z + (rz + 1) * sbs);

    sdf_vals[7] = get_sdf(bsize, r, bpos, make_uint3(rx, ry + 1, rz + 1), max_vec_num, indexer, cube_sdf, cube_std, vec_batch_mapping);
    if (isnan(sdf_vals[7].x))
        return;
    points[7] = make_float3(bpos.x + rx * sbs, bpos.y + (ry + 1) * sbs, bpos.z + (rz + 1) * sbs);

    // Find triangle config.
    int cube_type = 0;
    if (sdf_vals[0].x < 0)
        cube_type |= 1;
    if (sdf_vals[1].x < 0)
        cube_type |= 2;
    if (sdf_vals[2].x < 0)
        cube_type |= 4;
    if (sdf_vals[3].x < 0)
        cube_type |= 8;
    if (sdf_vals[4].x < 0)
        cube_type |= 16;
    if (sdf_vals[5].x < 0)
        cube_type |= 32;
    if (sdf_vals[6].x < 0)
        cube_type |= 64;
    if (sdf_vals[7].x < 0)
        cube_type |= 128;

    // Find vertex position on each edge (weighted by sdf value)
    int edge_config = edgeTable[cube_type];
    float4 vert_list[12];

    if (edge_config == 0)
        return;
    if (edge_config & 1)
        vert_list[0] = sdf_interp(points[0], points[1], sdf_vals[0].y, sdf_vals[1].y, sdf_vals[0].x, sdf_vals[1].x);
    if (edge_config & 2)
        vert_list[1] = sdf_interp(points[1], points[2], sdf_vals[1].y, sdf_vals[2].y, sdf_vals[1].x, sdf_vals[2].x);
    if (edge_config & 4)
        vert_list[2] = sdf_interp(points[2], points[3], sdf_vals[2].y, sdf_vals[3].y, sdf_vals[2].x, sdf_vals[3].x);
    if (edge_config & 8)
        vert_list[3] = sdf_interp(points[3], points[0], sdf_vals[3].y, sdf_vals[0].y, sdf_vals[3].x, sdf_vals[0].x);
    if (edge_config & 16)
        vert_list[4] = sdf_interp(points[4], points[5], sdf_vals[4].y, sdf_vals[5].y, sdf_vals[4].x, sdf_vals[5].x);
    if (edge_config & 32)
        vert_list[5] = sdf_interp(points[5], points[6], sdf_vals[5].y, sdf_vals[6].y, sdf_vals[5].x, sdf_vals[6].x);
    if (edge_config & 64)
        vert_list[6] = sdf_interp(points[6], points[7], sdf_vals[6].y, sdf_vals[7].y, sdf_vals[6].x, sdf_vals[7].x);
    if (edge_config & 128)
        vert_list[7] = sdf_interp(points[7], points[4], sdf_vals[7].y, sdf_vals[4].y, sdf_vals[7].x, sdf_vals[4].x);
    if (edge_config & 256)
        vert_list[8] = sdf_interp(points[0], points[4], sdf_vals[0].y, sdf_vals[4].y, sdf_vals[0].x, sdf_vals[4].x);
    if (edge_config & 512)
        vert_list[9] = sdf_interp(points[1], points[5], sdf_vals[1].y, sdf_vals[5].y, sdf_vals[1].x, sdf_vals[5].x);
    if (edge_config & 1024)
        vert_list[10] = sdf_interp(points[2], points[6], sdf_vals[2].y, sdf_vals[6].y, sdf_vals[2].x, sdf_vals[6].x);
    if (edge_config & 2048)
        vert_list[11] = sdf_interp(points[3], points[7], sdf_vals[3].y, sdf_vals[7].y, sdf_vals[3].x, sdf_vals[7].x);

    // Write triangles to array.
    float4 vp[3];
    for (int i = 0; triangleTable[cube_type][i] != -1; i += 3)
    {
#pragma unroll
        for (int vi = 0; vi < 3; ++vi)
        {
            vp[vi] = vert_list[triangleTable[cube_type][i + vi]];
        }
        if (vp[0].w > max_std || vp[1].w > max_std || vp[2].w > max_std)
        {
            continue;
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
                triangle_std[triangle_id][vi] = vp[vi].w;
            }
            triangle_flatten_id[triangle_id] = valid_blocks[lif_id];
        }
    }
}

std::vector<torch::Tensor> marching_cubes_sparse_interp_cuda(
    torch::Tensor indexer,           // (nx, ny, nz) -> data_id
    torch::Tensor valid_blocks,      // (K, )
    torch::Tensor vec_batch_mapping, //
    torch::Tensor cube_sdf,          // (M, rx, ry, rz)
    torch::Tensor cube_std,          // (M, rx, ry, rz)
    const std::vector<int> &n_xyz,   // [nx, ny, nz]
    float max_std,                   // Prune all vertices
    int max_n_triangles              // Maximum number of triangle buffer
)
{
    CHECK_INPUT(indexer);
    CHECK_INPUT(valid_blocks);
    CHECK_INPUT(cube_sdf);
    CHECK_INPUT(cube_std);
    CHECK_INPUT(vec_batch_mapping);
    assert(max_n_triangles > 0);

    const int r = cube_sdf.size(1) / 2;
    const int r3 = r * r * r;
    const int num_lif = valid_blocks.size(0);
    const uint max_vec_num = vec_batch_mapping.size(0);

    torch::Tensor triangles = torch::empty({max_n_triangles, 3, 3},
                                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor triangle_flatten_id = torch::empty({max_n_triangles}, torch::dtype(torch::kLong).device(torch::kCUDA));
    torch::Tensor triangle_std = torch::empty({max_n_triangles, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 dimBlock = dim3(16, 16);
    uint xBlocks = (num_lif + dimBlock.x - 1) / dimBlock.x;
    uint yBlocks = (r3 + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid = dim3(xBlocks, yBlocks);

    thrust::device_vector<int> n_output(1, 0);
    meshing_cube<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        indexer.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
        valid_blocks.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        vec_batch_mapping.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        cube_sdf.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        cube_std.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        triangles.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        triangle_std.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        triangle_flatten_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        n_output.data().get(), max_n_triangles, max_vec_num,
        n_xyz[0], n_xyz[1], n_xyz[2], max_std);
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

    int output_n_triangles = n_output[0];
    if (output_n_triangles < max_n_triangles)
    {
        // Trim output tensor if it is not full.
        triangles = triangles.index({torch::indexing::Slice(torch::indexing::None, output_n_triangles)});
        triangle_flatten_id = triangle_flatten_id.index({torch::indexing::Slice(torch::indexing::None, output_n_triangles)});
        triangle_std = triangle_std.index({torch::indexing::Slice(torch::indexing::None, output_n_triangles)});
    }
    else
    {
        // Otherwise spawn a warning.
        std::cerr << "Warning from marching cube: the max triangle number is too small " << output_n_triangles << " vs " << max_n_triangles << std::endl;
    }

    return {triangles, triangle_flatten_id, triangle_std};
}
