#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>

#define MAX_BITS 21
// #define SCALE_MASK ((uint64_t)0x1FF)
#define SCALE_MASK ((uint64_t)0x1)

template <class T>
struct Vector3
{
    Vector3() : x(0), y(0), z(0) {}
    Vector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    Vector3<T> operator+(const Vector3<T> &b)
    {
        return Vector3<T>(x + b.x, y + b.y, z + b.z);
    }

    Vector3<T> operator-(const Vector3<T> &b)
    {
        return Vector3<T>(x - b.x, y - b.y, z - b.z);
    }

    T x, y, z;
};

typedef Vector3<int> Vector3i;
typedef Vector3<float> Vector3f;

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

inline uint64_t expand(unsigned long long value)
{
    uint64_t x = value & 0x1fffff;
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

inline uint64_t compute_morton(uint64_t x, uint64_t y, uint64_t z)
{
    uint64_t code = 0;

    x = expand(x);
    y = expand(y) << 1;
    z = expand(z) << 2;

    code = x | y | z;
    return code;
}

inline Eigen::Vector3i decode(const uint64_t code)
{
    return {
        compact(code >> 0ull),
        compact(code >> 1ull),
        compact(code >> 2ull)};
}

inline uint64_t encode(const int x, const int y, const int z)
{
    return (compute_morton(x, y, z) & MASK[MAX_BITS - 1]);
}