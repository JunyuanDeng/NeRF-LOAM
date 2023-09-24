#pragma once
#include <iostream>

#define MAX_BITS 21
// #define SCALE_MASK ((uint64_t)0x1FF)
#define SCALE_MASK ((uint64_t)0x1)

/*
 * Mask generated with:
   MASK[0] = 0x7000000000000000,
   for(int i = 1; i < 21; ++i) {
   MASK[i] = MASK[i-1] | (MASK[0] >> (i*3));
   std::bitset<64> b(MASK[i]);
   std::cout << std::hex << b.to_ullong() << std::endl;
   }
 *
*/
constexpr uint64_t MASK[] = {
    0x7000000000000000,
    0x7e00000000000000,
    0x7fc0000000000000,
    0x7ff8000000000000,
    0x7fff000000000000,
    0x7fffe00000000000,
    0x7ffffc0000000000,
    0x7fffff8000000000,
    0x7ffffff000000000,
    0x7ffffffe00000000,
    0x7fffffffc0000000,
    0x7ffffffff8000000,
    0x7fffffffff000000,
    0x7fffffffffe00000,
    0x7ffffffffffc0000,
    0x7fffffffffff8000,
    0x7ffffffffffff000,
    0x7ffffffffffffe00,
    0x7fffffffffffffc0,
    0x7ffffffffffffff8,
    0x7fffffffffffffff};

inline int64_t expand(int64_t value)
{
    int64_t x = value & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline uint64_t compact(uint64_t value)
{
    uint64_t x = value & 0x1249249249249249;
    x = (x | x >> 2) & 0x10c30c30c30c30c3;
    x = (x | x >> 4) & 0x100f00f00f00f00f;
    x = (x | x >> 8) & 0x1f0000ff0000ff;
    x = (x | x >> 16) & 0x1f00000000ffff;
    x = (x | x >> 32) & 0x1fffff;
    return x;
}

inline int64_t compute_morton(int64_t x, int64_t y, int64_t z)
{
    int64_t code = 0;

    x = expand(x);
    y = expand(y) << 1;
    z = expand(z) << 2;

    code = x | y | z;
    return code;
}

inline torch::Tensor encode_torch(torch::Tensor coords)
{
    torch::Tensor outs = torch::zeros({coords.size(0), 1}, dtype(torch::kInt64));
    for (int i = 0; i < coords.size(0); ++i)
    {
        int64_t x = coords.data_ptr<int64_t>()[i * 3];
        int64_t y = coords.data_ptr<int64_t>()[i * 3 + 1];
        int64_t z = coords.data_ptr<int64_t>()[i * 3];
        outs.data_ptr<int64_t>()[i] = (compute_morton(x, y, z) & MASK[MAX_BITS - 1]);
    }
    return outs;
}
