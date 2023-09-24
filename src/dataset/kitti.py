import os.path as osp

import cv2
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
import sys
from scipy.spatial import cKDTree

patchwork_module_path ="/home/pl21n4/Programmes/patchwork-plusplus/build/python_wrapper"
sys.path.insert(0, patchwork_module_path)
import pypatchworkpp
params = pypatchworkpp.Parameters()

PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)


class DataLoader(Dataset):
    def __init__(self, data_path, use_gt=False, max_depth=-1, min_depth=-1) -> None:
        self.data_path = data_path
        self.num_bin = len(glob(osp.join(self.data_path, "velodyne/*.bin")))
        self.use_gt = use_gt
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.gt_pose = self.load_gt_pose() if use_gt else None

    def get_init_pose(self, frame):
        if self.gt_pose is not None:
            return np.concatenate((self.gt_pose[frame], [0, 0, 0, 1])
                                  ).reshape(4, 4)
        else:
            return np.eye(4)

    def load_gt_pose(self):
        gt_file = osp.join(self.data_path, "poses_lidar.txt")
        gt_pose = np.loadtxt(gt_file)
        return gt_pose

    def load_points(self, index):
        remove_abnormal_z = True
        path = osp.join(self.data_path, "velodyne/{:06d}.bin".format(index))
        points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
        if remove_abnormal_z:
            points = points[points[:, 2] > -3.0]
        points_norm = np.linalg.norm(points[:, :3], axis=-1)
        point_mask = True
        if self.max_depth != -1:
            point_mask = (points_norm < self.max_depth) & point_mask
        if self.min_depth != -1:
            point_mask = (points_norm > self.min_depth) & point_mask

        if isinstance(point_mask, np.ndarray):
            points = points[point_mask]

        PatchworkPLUSPLUS.estimateGround(points)
        ground = PatchworkPLUSPLUS.getGround()
        nonground = PatchworkPLUSPLUS.getNonground()
        Patchcenters = PatchworkPLUSPLUS.getCenters()
        normals = PatchworkPLUSPLUS.getNormals()
        T = cKDTree(Patchcenters)
        _, index = T.query(ground)
        if True:
            groundcos = np.abs(np.sum(normals[index] * ground, axis=-1)/np.linalg.norm(ground, axis=-1))
        else:
            groundcos = np.ones(ground.shape[0])
        points = np.concatenate((ground, nonground), axis=0)
        pointcos = np.concatenate((groundcos, np.ones(nonground.shape[0])), axis=0)

        return points, pointcos

    def __len__(self):
        return self.num_bin

    def __getitem__(self, index):
        points, pointcos = self.load_points(index)
        points = torch.from_numpy(points).float()
        pointcos = torch.from_numpy(pointcos).float()
        pose = np.concatenate((self.gt_pose[index], [0, 0, 0, 1])
                              ).reshape(4, 4) if self.use_gt else None
        return index, points, pointcos, pose


if __name__ == "__main__":
    path = "/home/pl21n4/dataset/kitti/dataset/sequences/00/"
    loader = DataLoader(path)
    for data in loader:
        index, points, pose = data
        print("current index ", index)
        print("first 10th points:\n", points[:10])
        if index > 10:
            break
        index += 1
