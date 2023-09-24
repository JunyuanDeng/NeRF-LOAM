import math
import torch

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from variations.render_helpers import get_scores, eval_points


class MeshExtractor:
    def __init__(self, args):
        self.voxel_size = args.mapper_specs["voxel_size"]
        self.rays_d = None
        self.depth_points = None

    @ torch.no_grad()
    def linearize_id(self, xyz, n_xyz):
        return xyz[:, 2] + n_xyz[-1] * xyz[:, 1] + (n_xyz[-1] * n_xyz[-2]) * xyz[:, 0]

    @torch.no_grad()
    def downsample_points(self, points, voxel_size=0.01):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd.points)

    @torch.no_grad()
    def get_rays(self, w=None, h=None, K=None):
        w = self.w if w == None else w
        h = self.h if h == None else h
        if K is None:
            K = np.eye(3)
            K[0, 0] = self.K[0, 0] * w / self.w
            K[1, 1] = self.K[1, 1] * h / self.h
            K[0, 2] = self.K[0, 2] * w / self.w
            K[1, 2] = self.K[1, 2] * h / self.h
        ix, iy = torch.meshgrid(
            torch.arange(w), torch.arange(h), indexing='xy')
        rays_d = torch.stack(
            [(ix-K[0, 2]) / K[0, 0],
             (iy-K[1, 2]) / K[1, 1],
             torch.ones_like(ix)], -1).float()
        return rays_d

    @torch.no_grad()
    def get_valid_points(self, frame_poses, depth_maps):
        if isinstance(frame_poses, list):
            all_points = []
            print("extracting all points")
            for i in range(0, len(frame_poses), 5):
                pose = frame_poses[i]
                depth = depth_maps[i]
                points = self.rays_d * depth.unsqueeze(-1)
                points = points.reshape(-1, 3)
                points = points @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
                if len(all_points) == 0:
                    all_points = points.detach().cpu().numpy()
                else:
                    all_points = np.concatenate(
                        [all_points, points.detach().cpu().numpy()], 0)
            print("downsample all points")
            all_points = self.downsample_points(all_points)
            return all_points
        else:
            pose = frame_poses
            depth = depth_maps
            points = self.rays_d * depth.unsqueeze(-1)
            points = points.reshape(-1, 3)
            points = points @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
            if self.depth_points is None:
                self.depth_points = points.detach().cpu().numpy()
            else:
                self.depth_points = np.concatenate(
                    [self.depth_points, points], 0)
            self.depth_points = self.downsample_points(self.depth_points)
        return self.depth_points

    @ torch.no_grad()
    def create_mesh(self, decoder, map_states, voxel_size, voxels,
                    frame_poses=None, depth_maps=None, clean_mseh=False,
                    require_color=False, offset=-80, res=8):

        sdf_grid = get_scores(decoder, map_states, voxel_size, bits=res)
        sdf_grid = sdf_grid.reshape(-1, res, res, res, 1)

        voxel_centres = map_states["voxel_center_xyz"]
        verts, faces = self.marching_cubes(voxel_centres, sdf_grid)

        if clean_mseh:
            print("********** get points from frames **********")
            all_points = self.get_valid_points(frame_poses, depth_maps)
            print("********** construct kdtree **********")
            kdtree = cKDTree(all_points)
            print("********** query kdtree **********")
            point_mask = kdtree.query_ball_point(
                verts, voxel_size * 0.5, workers=12, return_length=True)
            print("********** finished querying kdtree **********")
            point_mask = point_mask > 0
            face_mask = point_mask[faces.reshape(-1)].reshape(-1, 3).any(-1)

            faces = faces[face_mask]

        if require_color:
            print("********** get color from network **********")
            verts_torch = torch.from_numpy(verts).float().cuda()
            batch_points = torch.split(verts_torch, 1000)
            colors = []
            for points in batch_points:
                voxel_pos = points // self.voxel_size
                batch_voxels = voxels[:, :3].cuda()
                batch_voxels = batch_voxels.unsqueeze(
                    0).repeat(voxel_pos.shape[0], 1, 1)

                # filter outliers
                nonzeros = (batch_voxels == voxel_pos.unsqueeze(1)).all(-1)
                nonzeros = torch.where(nonzeros, torch.ones_like(
                    nonzeros).int(), -torch.ones_like(nonzeros).int())
                sorted, index = torch.sort(nonzeros, dim=-1, descending=True)
                sorted = sorted[:, 0]
                index = index[:, 0]
                valid = (sorted != -1)
                color_empty = torch.zeros_like(points)
                points = points[valid, :]
                index = index[valid]

                # get color
                if len(points) > 0:
                    color = eval_points(decoder, map_states,
                                        points, index, voxel_size).cuda()
                    color_empty[valid] = color
                colors += [color_empty]
            colors = torch.cat(colors, 0)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts+offset)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        if require_color:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                colors.detach().cpu().numpy())
        mesh.compute_vertex_normals()
        return mesh

    @ torch.no_grad()
    def marching_cubes(self, voxels, sdf):
        voxels = voxels[:, :3]
        sdf = sdf[..., 0]
        res = 1.0 / (sdf.shape[1] - 1)
        spacing = [res, res, res]

        num_verts = 0
        total_verts = []
        total_faces = []
        for i in range(len(voxels)):
            sdf_volume = sdf[i].detach().cpu().numpy()
            if np.min(sdf_volume) > 0 or np.max(sdf_volume) < 0:
                continue
            verts, faces, _, _ = marching_cubes(sdf_volume, 0, spacing=spacing)
            verts -= 0.5
            verts *= self.voxel_size
            verts += voxels[i].detach().cpu().numpy()
            faces += num_verts
            num_verts += verts.shape[0]

            total_verts += [verts]
            total_faces += [faces]
        total_verts = np.concatenate(total_verts)
        total_faces = np.concatenate(total_faces)
        return total_verts, total_faces
