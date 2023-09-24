import torch
import torch.nn as nn
import numpy as np
from se3pose import OptimizablePose
from utils.sample_util import *
import random


class LidarFrame(nn.Module):
    def __init__(self, index, points, pointsCos, pose=None, new_keyframe=False) -> None:
        super().__init__()
        self.index = index
        self.num_point = len(points)
        self.points = points
        self.pointsCos = pointsCos
        if (not new_keyframe) and (pose is not None):
            # TODO: fix this offset
            pose[:3, 3] += 2000
            pose = torch.tensor(pose, requires_grad=True, dtype=torch.float32)
            self.pose = OptimizablePose.from_matrix(pose)
        elif new_keyframe:
            self.pose = pose
        self.rays_d = self.get_rays()
        self.rel_pose = None

    def get_pose(self):
        return self.pose.matrix()

    def get_translation(self):
        return self.pose.translation()

    def get_rotation(self):
        return self.pose.rotation()

    def get_points(self):
        return self.points

    def get_pointsCos(self):
        return self.pointsCos

    def set_rel_pose(self, rel_pose):
        self.rel_pose = rel_pose

    def get_rel_pose(self):
        return self.rel_pose

    @torch.no_grad()
    def get_rays(self):
        self.rays_norm = (torch.norm(self.points, 2, -1, keepdim=True)+1e-8)
        rays_d = self.points / self.rays_norm
        # TODO: to keep cosistency, add one dim, but actually no need
        return rays_d.unsqueeze(1).float()

    @torch.no_grad()
    def sample_rays(self, N_rays, track=False):
        self.sample_mask = sample_rays(
            torch.ones((self.num_point, 1))[None, ...], N_rays)[0, ...]


