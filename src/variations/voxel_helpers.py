# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch """
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import sys
import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
import sys
import numpy as np
import grid as _ext

MAX_DEPTH = 80


class BallRayIntersect(Function):
    @staticmethod
    def forward(ctx, radius, n_max, points, ray_start, ray_dir):
        inds, min_depth, max_depth = _ext.ball_intersect(
            ray_start.float(), ray_dir.float(), points.float(), radius, n_max
        )
        min_depth = min_depth.type_as(ray_start)
        max_depth = max_depth.type_as(ray_start)

        ctx.mark_non_differentiable(inds)
        ctx.mark_non_differentiable(min_depth)
        ctx.mark_non_differentiable(max_depth)
        return inds, min_depth, max_depth

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None


ball_ray_intersect = BallRayIntersect.apply


class AABBRayIntersect(Function):
    @staticmethod
    def forward(ctx, voxelsize, n_max, points, ray_start, ray_dir):
        # HACK: speed-up ray-voxel intersection by batching...
        # HACK: avoid out-of-memory
        G = min(2048, int(2 * 10 ** 9 / points.numel()))
        S, N = ray_start.shape[:2]
        K = int(np.ceil(N / G))
        H = K * G
        if H > N:
            ray_start = torch.cat([ray_start, ray_start[:, : H - N]], 1)
            ray_dir = torch.cat([ray_dir, ray_dir[:, : H - N]], 1)
        ray_start = ray_start.reshape(S * G, K, 3)
        ray_dir = ray_dir.reshape(S * G, K, 3)
        points = points.expand(S * G, *points.size()[1:]).contiguous()

        inds, min_depth, max_depth = _ext.aabb_intersect(
            ray_start.float(), ray_dir.float(), points.float(), voxelsize, n_max
        )
        min_depth = min_depth.type_as(ray_start)
        max_depth = max_depth.type_as(ray_start)

        inds = inds.reshape(S, H, -1)
        min_depth = min_depth.reshape(S, H, -1)
        max_depth = max_depth.reshape(S, H, -1)
        if H > N:
            inds = inds[:, :N]
            min_depth = min_depth[:, :N]
            max_depth = max_depth[:, :N]

        ctx.mark_non_differentiable(inds)
        ctx.mark_non_differentiable(min_depth)
        ctx.mark_non_differentiable(max_depth)
        return inds, min_depth, max_depth

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None


aabb_ray_intersect = AABBRayIntersect.apply


class SparseVoxelOctreeRayIntersect(Function):
    @staticmethod
    def forward(ctx, voxelsize, n_max, points, children, ray_start, ray_dir):
        # HACK: avoid out-of-memory
        torch.cuda.empty_cache()
        G = min(256, int(2 * 10 ** 9 / (points.numel() + children.numel())))
        S, N = ray_start.shape[:2]
        K = int(np.ceil(N / G))
        H = K * G
        if H > N:
            ray_start = torch.cat([ray_start, ray_start[:, : H - N]], 1)
            ray_dir = torch.cat([ray_dir, ray_dir[:, : H - N]], 1)
        ray_start = ray_start.reshape(S * G, K, 3)
        ray_dir = ray_dir.reshape(S * G, K, 3)
        points = points.expand(S * G, *points.size()).contiguous()
        torch.cuda.empty_cache()
        children = children.expand(S * G, *children.size()).contiguous()
        torch.cuda.empty_cache()
        inds, min_depth, max_depth = _ext.svo_intersect(
            ray_start.float(),
            ray_dir.float(),
            points.float(),
            children.int(),
            voxelsize,
            n_max,
        )
        torch.cuda.empty_cache()
        min_depth = min_depth.type_as(ray_start)
        max_depth = max_depth.type_as(ray_start)

        inds = inds.reshape(S, H, -1)
        min_depth = min_depth.reshape(S, H, -1)
        max_depth = max_depth.reshape(S, H, -1)
        if H > N:
            inds = inds[:, :N]
            min_depth = min_depth[:, :N]
            max_depth = max_depth[:, :N]

        ctx.mark_non_differentiable(inds)
        ctx.mark_non_differentiable(min_depth)
        ctx.mark_non_differentiable(max_depth)
        return inds, min_depth, max_depth

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None


svo_ray_intersect = SparseVoxelOctreeRayIntersect.apply


class TriangleRayIntersect(Function):
    @staticmethod
    def forward(ctx, cagesize, blur_ratio, n_max, points, faces, ray_start, ray_dir):
        # HACK: speed-up ray-voxel intersection by batching...
        # HACK: avoid out-of-memory
        G = min(2048, int(2 * 10 ** 9 / (3 * faces.numel())))
        S, N = ray_start.shape[:2]
        K = int(np.ceil(N / G))
        H = K * G
        if H > N:
            ray_start = torch.cat([ray_start, ray_start[:, : H - N]], 1)
            ray_dir = torch.cat([ray_dir, ray_dir[:, : H - N]], 1)
        ray_start = ray_start.reshape(S * G, K, 3)
        ray_dir = ray_dir.reshape(S * G, K, 3)
        face_points = F.embedding(faces.reshape(-1, 3), points.reshape(-1, 3))
        face_points = (
            face_points.unsqueeze(0).expand(
                S * G, *face_points.size()).contiguous()
        )
        inds, depth, uv = _ext.triangle_intersect(
            ray_start.float(),
            ray_dir.float(),
            face_points.float(),
            cagesize,
            blur_ratio,
            n_max,
        )
        depth = depth.type_as(ray_start)
        uv = uv.type_as(ray_start)

        inds = inds.reshape(S, H, -1)
        depth = depth.reshape(S, H, -1, 3)
        uv = uv.reshape(S, H, -1)
        if H > N:
            inds = inds[:, :N]
            depth = depth[:, :N]
            uv = uv[:, :N]

        ctx.mark_non_differentiable(inds)
        ctx.mark_non_differentiable(depth)
        ctx.mark_non_differentiable(uv)
        return inds, depth, uv

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None, None


triangle_ray_intersect = TriangleRayIntersect.apply


class UniformRaySampling(Function):
    @staticmethod
    def forward(
        ctx,
        pts_idx,
        min_depth,
        max_depth,
        step_size,
        max_ray_length,
        deterministic=False,
    ):
        G, N, P = 256, pts_idx.size(0), pts_idx.size(1)
        H = int(np.ceil(N / G)) * G
        if H > N:
            pts_idx = torch.cat([pts_idx, pts_idx[: H - N]], 0)
            min_depth = torch.cat([min_depth, min_depth[: H - N]], 0)
            max_depth = torch.cat([max_depth, max_depth[: H - N]], 0)
        pts_idx = pts_idx.reshape(G, -1, P)
        min_depth = min_depth.reshape(G, -1, P)
        max_depth = max_depth.reshape(G, -1, P)

        # pre-generate noise
        max_steps = int(max_ray_length / step_size)
        max_steps = max_steps + min_depth.size(-1) * 2
        noise = min_depth.new_zeros(*min_depth.size()[:-1], max_steps)
        if deterministic:
            noise += 0.5
        else:
            noise = noise.uniform_()

        # call cuda function
        sampled_idx, sampled_depth, sampled_dists = _ext.uniform_ray_sampling(
            pts_idx,
            min_depth.float(),
            max_depth.float(),
            noise.float(),
            step_size,
            max_steps,
        )
        sampled_depth = sampled_depth.type_as(min_depth)
        sampled_dists = sampled_dists.type_as(min_depth)

        sampled_idx = sampled_idx.reshape(H, -1)
        sampled_depth = sampled_depth.reshape(H, -1)
        sampled_dists = sampled_dists.reshape(H, -1)
        if H > N:
            sampled_idx = sampled_idx[:N]
            sampled_depth = sampled_depth[:N]
            sampled_dists = sampled_dists[:N]

        max_len = sampled_idx.ne(-1).sum(-1).max()
        sampled_idx = sampled_idx[:, :max_len]
        sampled_depth = sampled_depth[:, :max_len]
        sampled_dists = sampled_dists[:, :max_len]

        ctx.mark_non_differentiable(sampled_idx)
        ctx.mark_non_differentiable(sampled_depth)
        ctx.mark_non_differentiable(sampled_dists)
        return sampled_idx, sampled_depth, sampled_dists

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None, None


uniform_ray_sampling = UniformRaySampling.apply


class InverseCDFRaySampling(Function):
    @staticmethod
    def forward(
        ctx,
        pts_idx,
        min_depth,
        max_depth,
        probs,
        steps,
        fixed_step_size=-1,
        deterministic=False,
    ):
        G, N, P = 200, pts_idx.size(0), pts_idx.size(1)
        H = int(np.ceil(N / G)) * G

        if H > N:
            pts_idx = torch.cat([pts_idx, pts_idx[:1].expand(H - N, P)], 0)
            min_depth = torch.cat(
                [min_depth, min_depth[:1].expand(H - N, P)], 0)
            max_depth = torch.cat(
                [max_depth, max_depth[:1].expand(H - N, P)], 0)
            probs = torch.cat([probs, probs[:1].expand(H - N, P)], 0)
            steps = torch.cat([steps, steps[:1].expand(H - N)], 0)

        # print(G, P, np.ceil(N / G), N, H, pts_idx.shape, min_depth.device)
        pts_idx = pts_idx.reshape(G, -1, P)
        min_depth = min_depth.reshape(G, -1, P)
        max_depth = max_depth.reshape(G, -1, P)
        probs = probs.reshape(G, -1, P)
        steps = steps.reshape(G, -1)

        # pre-generate noise
        max_steps = steps.ceil().long().max() + P
        # print(max_steps)
        # print(*min_depth.size()[:-1]," ", max_steps)
        noise = min_depth.new_zeros(*min_depth.size()[:-1], max_steps)
        if deterministic:
            noise += 0.5
        else:
            noise = noise.uniform_().clamp(min=0.001, max=0.999)  # in case

        # call cuda function
        chunk_size = 4 * G  # to avoid oom?
        results = [
            _ext.inverse_cdf_sampling(
                pts_idx[:, i: i + chunk_size].contiguous(),
                min_depth.float()[:, i: i + chunk_size].contiguous(),
                max_depth.float()[:, i: i + chunk_size].contiguous(),
                noise.float()[:, i: i + chunk_size].contiguous(),
                probs.float()[:, i: i + chunk_size].contiguous(),
                steps.float()[:, i: i + chunk_size].contiguous(),
                fixed_step_size,
            )
            for i in range(0, min_depth.size(1), chunk_size)
        ]

        sampled_idx, sampled_depth, sampled_dists = [
            torch.cat([r[i] for r in results], 1) for i in range(3)
        ]
        sampled_depth = sampled_depth.type_as(min_depth)
        sampled_dists = sampled_dists.type_as(min_depth)

        sampled_idx = sampled_idx.reshape(H, -1)
        sampled_depth = sampled_depth.reshape(H, -1)
        sampled_dists = sampled_dists.reshape(H, -1)
        if H > N:
            sampled_idx = sampled_idx[:N]
            sampled_depth = sampled_depth[:N]
            sampled_dists = sampled_dists[:N]

        max_len = sampled_idx.ne(-1).sum(-1).max()
        sampled_idx = sampled_idx[:, :max_len]
        sampled_depth = sampled_depth[:, :max_len]
        sampled_dists = sampled_dists[:, :max_len]

        ctx.mark_non_differentiable(sampled_idx)
        ctx.mark_non_differentiable(sampled_depth)
        ctx.mark_non_differentiable(sampled_dists)
        return sampled_idx, sampled_depth, sampled_dists

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None, None, None


inverse_cdf_sampling = InverseCDFRaySampling.apply


# back-up for ray point sampling
@torch.no_grad()
def _parallel_ray_sampling(
    MARCH_SIZE, pts_idx, min_depth, max_depth, deterministic=False
):
    # uniform sampling
    _min_depth = min_depth.min(1)[0]
    _max_depth = max_depth.masked_fill(max_depth.eq(MAX_DEPTH), 0).max(1)[0]
    max_ray_length = (_max_depth - _min_depth).max()

    delta = torch.arange(
        int(max_ray_length / MARCH_SIZE), device=min_depth.device, dtype=min_depth.dtype
    )
    delta = delta[None, :].expand(min_depth.size(0), delta.size(-1))
    if deterministic:
        delta = delta + 0.5
    else:
        delta = delta + delta.clone().uniform_().clamp(min=0.01, max=0.99)
    delta = delta * MARCH_SIZE
    sampled_depth = min_depth[:, :1] + delta
    sampled_idx = (sampled_depth[:, :, None] >=
                   min_depth[:, None, :]).sum(-1) - 1
    sampled_idx = pts_idx.gather(1, sampled_idx)

    # include all boundary points
    sampled_depth = torch.cat([min_depth, max_depth, sampled_depth], -1)
    sampled_idx = torch.cat([pts_idx, pts_idx, sampled_idx], -1)

    # reorder
    sampled_depth, ordered_index = sampled_depth.sort(-1)
    sampled_idx = sampled_idx.gather(1, ordered_index)
    sampled_dists = sampled_depth[:, 1:] - sampled_depth[:, :-1]  # distances
    sampled_depth = 0.5 * \
        (sampled_depth[:, 1:] + sampled_depth[:, :-1])  # mid-points

    # remove all invalid depths
    min_ids = (sampled_depth[:, :, None] >= min_depth[:, None, :]).sum(-1) - 1
    max_ids = (sampled_depth[:, :, None] >= max_depth[:, None, :]).sum(-1)

    sampled_depth.masked_fill_(
        (max_ids.ne(min_ids))
        | (sampled_depth > _max_depth[:, None])
        | (sampled_dists == 0.0),
        MAX_DEPTH,
    )
    sampled_depth, ordered_index = sampled_depth.sort(-1)  # sort again
    sampled_masks = sampled_depth.eq(MAX_DEPTH)
    num_max_steps = (~sampled_masks).sum(-1).max()

    sampled_depth = sampled_depth[:, :num_max_steps]
    sampled_dists = sampled_dists.gather(1, ordered_index).masked_fill_(
        sampled_masks, 0.0
    )[:, :num_max_steps]
    sampled_idx = sampled_idx.gather(1, ordered_index).masked_fill_(sampled_masks, -1)[
        :, :num_max_steps
    ]

    return sampled_idx, sampled_depth, sampled_dists


@torch.no_grad()
def parallel_ray_sampling(
    MARCH_SIZE, pts_idx, min_depth, max_depth, deterministic=False
):
    chunk_size = 4096
    full_size = min_depth.shape[0]
    if full_size <= chunk_size:
        return _parallel_ray_sampling(
            MARCH_SIZE, pts_idx, min_depth, max_depth, deterministic=deterministic
        )

    outputs = zip(
        *[
            _parallel_ray_sampling(
                MARCH_SIZE,
                pts_idx[i: i + chunk_size],
                min_depth[i: i + chunk_size],
                max_depth[i: i + chunk_size],
                deterministic=deterministic,
            )
            for i in range(0, full_size, chunk_size)
        ]
    )
    sampled_idx, sampled_depth, sampled_dists = outputs

    def padding_points(xs, pad):
        if len(xs) == 1:
            return xs[0]

        maxlen = max([x.size(1) for x in xs])
        full_size = sum([x.size(0) for x in xs])
        xt = xs[0].new_ones(full_size, maxlen).fill_(pad)
        st = 0
        for i in range(len(xs)):
            xt[st: st + xs[i].size(0), : xs[i].size(1)] = xs[i]
            st += xs[i].size(0)
        return xt

    sampled_idx = padding_points(sampled_idx, -1)
    sampled_depth = padding_points(sampled_depth, MAX_DEPTH)
    sampled_dists = padding_points(sampled_dists, 0.0)
    return sampled_idx, sampled_depth, sampled_dists


def discretize_points(voxel_points, voxel_size):
    # this function turns voxel centers/corners into integer indeices
    # we assume all points are alreay put as voxels (real numbers)
    minimal_voxel_point = voxel_points.min(dim=0, keepdim=True)[0]
    voxel_indices = (
        ((voxel_points - minimal_voxel_point) / voxel_size).round_().long()
    )  # float
    residual = (voxel_points - voxel_indices.type_as(voxel_points) * voxel_size).mean(
        0, keepdim=True
    )
    return voxel_indices, residual


def build_easy_octree(points, half_voxel):
    coords, residual = discretize_points(points, half_voxel)
    ranges = coords.max(0)[0] - coords.min(0)[0]
    depths = torch.log2(ranges.max().float()).ceil_().long() - 1
    center = (coords.max(0)[0] + coords.min(0)[0]) / 2
    centers, children = _ext.build_octree(center, coords, int(depths))
    centers = centers.float() * half_voxel + residual  # transform back to float
    return centers, children


@torch.enable_grad()
def trilinear_interp(p, q, point_feats):
    weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)
    if point_feats.dim() == 2:
        point_feats = point_feats.view(point_feats.size(0), 8, -1)

    point_feats = (weights * point_feats).sum(1)
    return point_feats


def offset_points(point_xyz, quarter_voxel=1, offset_only=False, bits=2):
    c = torch.arange(1, 2 * bits, 2, device=point_xyz.device)
    ox, oy, oz = torch.meshgrid([c, c, c], indexing='ij')
    offset = (torch.cat([ox.reshape(-1, 1),
                         oy.reshape(-1, 1),
                         oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
    if not offset_only:
        return (
            point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel)
    return offset.type_as(point_xyz) * quarter_voxel


def splitting_points(point_xyz, point_feats, values, half_voxel):
    # generate new centers
    quarter_voxel = half_voxel * 0.5
    new_points = offset_points(point_xyz, quarter_voxel).reshape(-1, 3)
    old_coords = discretize_points(point_xyz, quarter_voxel)[0]
    new_coords = offset_points(old_coords).reshape(-1, 3)
    new_keys0 = offset_points(new_coords).reshape(-1, 3)

    # get unique keys and inverse indices (for original key0, where it maps to in keys)
    new_keys, new_feats = torch.unique(
        new_keys0, dim=0, sorted=True, return_inverse=True)
    new_keys_idx = new_feats.new_zeros(new_keys.size(0)).scatter_(
        0, new_feats, torch.arange(new_keys0.size(0), device=new_feats.device) // 64)

    # recompute key vectors using trilinear interpolation
    new_feats = new_feats.reshape(-1, 8)

    if values is not None:
        # (1/4 voxel size)
        p = (new_keys - old_coords[new_keys_idx]
             ).type_as(point_xyz).unsqueeze(1) * 0.25 + 0.5
        q = offset_points(p, 0.5, offset_only=True).unsqueeze(0) + 0.5  # BUG?
        point_feats = point_feats[new_keys_idx]
        point_feats = F.embedding(point_feats, values).view(
            point_feats.size(0), -1)
        new_values = trilinear_interp(p, q, point_feats)
    else:
        new_values = None
    return new_points, new_feats, new_values, new_keys


@torch.no_grad()
def ray_intersect(ray_start, ray_dir, flatten_centers, flatten_children, voxel_size, max_hits, max_distance=MAX_DEPTH):
    # ray-voxel intersection
    max_hits_temp = 20
    pts_idx, min_depth, max_depth = svo_ray_intersect(
        voxel_size,
        max_hits_temp,
        flatten_centers,
        flatten_children,
        ray_start,
        ray_dir)
    torch.cuda.empty_cache()
    # sort the depths
    min_depth.masked_fill_(pts_idx.eq(-1), max_distance)
    max_depth.masked_fill_(pts_idx.eq(-1), max_distance)

    min_depth, sorted_idx = min_depth.sort(dim=-1)
    max_depth = max_depth.gather(-1, sorted_idx)
    pts_idx = pts_idx.gather(-1, sorted_idx)
    # print(max_depth.max())
    pts_idx[max_depth > 2*max_distance] = -1
    pts_idx[min_depth > max_distance] = -1
    min_depth.masked_fill_(pts_idx.eq(-1), max_distance)
    max_depth.masked_fill_(pts_idx.eq(-1), max_distance)
    # remove all points that completely miss the object
    max_hits = torch.max(pts_idx.ne(-1).sum(-1))
    min_depth = min_depth[..., :max_hits]
    max_depth = max_depth[..., :max_hits]
    pts_idx = pts_idx[..., :max_hits]

    hits = pts_idx.ne(-1).any(-1)

    intersection_outputs = {
        "min_depth": min_depth,
        "max_depth": max_depth,
        "intersected_voxel_idx": pts_idx,
    }
    return intersection_outputs, hits


@torch.no_grad()
def ray_sample(intersection_outputs, step_size=0.01, fixed=False):
    dists = (
        intersection_outputs["max_depth"] -
        intersection_outputs["min_depth"]
    ).masked_fill(intersection_outputs["intersected_voxel_idx"].eq(-1), 0)
    intersection_outputs["probs"] = dists / dists.sum(dim=-1, keepdim=True)
    intersection_outputs["steps"] = dists.sum(-1) / step_size
    # TODO:A serious BUG need to fix!
    if dists.sum(-1).max() > 10 * MAX_DEPTH:
        return
    # sample points and use middle point approximation
    sampled_idx, sampled_depth, sampled_dists = inverse_cdf_sampling(
        intersection_outputs["intersected_voxel_idx"],
        intersection_outputs["min_depth"],
        intersection_outputs["max_depth"],
        intersection_outputs["probs"],
        intersection_outputs["steps"], -1, fixed)

    sampled_dists = sampled_dists.clamp(min=0.0)
    sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
    sampled_dists.masked_fill_(sampled_idx.eq(-1), 0.0)

    samples = {
        "sampled_point_depth": sampled_depth,
        "sampled_point_distance": sampled_dists,
        "sampled_point_voxel_idx": sampled_idx,
    }
    return samples
