import numpy as np


def compact(value):
    x = value & 0x1249249249249249
    x = (x | x >> 2) & 0x10c30c30c30c30c3
    x = (x | x >> 4) & 0x100f00f00f00f00f
    x = (x | x >> 8) & 0x1f0000ff0000ff
    x = (x | x >> 16) & 0x1f00000000ffff
    x = (x | x >> 32) & 0x1fffff
    return x


def decode(code):
    return compact(code >> 0), compact(code >> 1), compact(code >> 2)


for i in range(10):
    x, y, z = decode(samples_valid['sampled_point_voxel_idx'][i])
    print(x, y, z)
    print(torch.sqrt((x-80)**2+(y-80)**2+(z-80)**2))
    print(samples_valid['sampled_point_depth'][i])
