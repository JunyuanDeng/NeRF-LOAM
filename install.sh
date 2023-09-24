#!/bin/bash

cd third_party/marching_cubes
python setup.py install

cd ../sparse_octree
python setup.py install

cd ../sparse_voxels
python setup.py install