from copy import deepcopy
import random
from time import sleep
import numpy as np
from tqdm import tqdm
import torch

from criterion import Criterion
from loggers import BasicLogger
from utils.import_util import get_decoder, get_property
from variations.render_helpers import bundle_adjust_frames
from utils.mesh_util import MeshExtractor
from utils.profile_util import Profiler
from lidarFrame import LidarFrame
import torch.nn.functional as F
from pathlib import Path
import open3d as o3d

torch.classes.load_library(
    "/home/pl21n4/Programmes/Vox-Fusion/third_party/sparse_octree/build/lib.linux-x86_64-cpython-38/svo.cpython-38-x86_64-linux-gnu.so")


def get_network_size(net):
    size = 0
    for param in net.parameters():
        size += param.element_size() * param.numel()
    return size / 1024 / 1024


class Mapping:
    def __init__(self, args, logger: BasicLogger):
        super().__init__()
        self.args = args
        self.logger = logger
        self.decoder = get_decoder(args).cuda()
        print(self.decoder)
        self.loss_criteria = Criterion(args)
        self.keyframe_graph = []
        self.initialized = False

        mapper_specs = args.mapper_specs

        # optional args
        self.ckpt_freq = get_property(args, "ckpt_freq", -1)
        self.final_iter = get_property(mapper_specs, "final_iter", False)
        self.mesh_res = get_property(mapper_specs, "mesh_res", 8)
        self.save_data_freq = get_property(
            args.debug_args, "save_data_freq", 0)

        # required args
        self.voxel_size = mapper_specs["voxel_size"]
        self.window_size = mapper_specs["window_size"]
        self.num_iterations = mapper_specs["num_iterations"]
        self.n_rays = mapper_specs["N_rays_each"]
        self.sdf_truncation = args.criteria["sdf_truncation"]
        self.max_voxel_hit = mapper_specs["max_voxel_hit"]
        self.step_size = mapper_specs["step_size"]
        self.learning_rate_emb = mapper_specs["learning_rate_emb"]
        self.learning_rate_decorder = mapper_specs["learning_rate_decorder"]
        self.learning_rate_pose = mapper_specs["learning_rate_pose"]
        self.step_size = self.step_size * self.voxel_size
        self.max_distance = args.data_specs["max_depth"]
        self.freeze_frame = mapper_specs["freeze_frame"]
        self.keyframe_gap = mapper_specs["keyframe_gap"]
        self.remove_back = mapper_specs["remove_back"]
        self.key_distance = mapper_specs["key_distance"]
        
        embed_dim = args.decoder_specs["in_dim"]
        use_local_coord = mapper_specs["use_local_coord"]
        self.embed_dim = embed_dim - 3 if use_local_coord else embed_dim
        #num_embeddings = mapper_specs["num_embeddings"]
        self.mesh_freq = args.debug_args["mesh_freq"]
        self.mesher = MeshExtractor(args)


        self.voxel_id2embedding_id = -torch.ones((int(2e9), 1), dtype=torch.int)
        self.embeds_exist_search = dict()
        self.current_num_embeds = 0
        self.dynamic_embeddings = None

        self.svo = torch.classes.svo.Octree()
        self.svo.init(256*256*4, embed_dim, self.voxel_size)

        self.frame_poses = []
        self.depth_maps = []
        self.last_tracked_frame_id = 0
        self.final_poses=[]
        
        verbose = get_property(args.debug_args, "verbose", False)
        self.profiler = Profiler(verbose=verbose)
        self.profiler.enable()
        
    def spin(self, share_data, kf_buffer):
        print("mapping process started!!!!!!!!!")
        while True:
            torch.cuda.empty_cache()
            if not kf_buffer.empty():
                tracked_frame = kf_buffer.get()
                # self.create_voxels(tracked_frame)
                if not self.initialized:
                    self.first_frame_id = tracked_frame.index
                    if self.mesher is not None:
                        self.mesher.rays_d = tracked_frame.get_rays()
                    self.create_voxels(tracked_frame)
                    self.insert_keyframe(tracked_frame)
                    while kf_buffer.empty():
                        self.do_mapping(share_data, tracked_frame, selection_method='current')
                    self.initialized = True
                else:
                    if self.remove_back:
                        tracked_frame = self.remove_back_points(tracked_frame)
                    self.do_mapping(share_data, tracked_frame)
                    self.create_voxels(tracked_frame)
                    if (torch.norm(tracked_frame.pose.translation().cpu()
                        - self.current_keyframe.pose.translation().cpu())) > self.keyframe_gap:
                        self.insert_keyframe(tracked_frame)
                        print(
                            f"********** current num kfs: { len(self.keyframe_graph) } **********")

                # self.create_voxels(tracked_frame)
                tracked_pose = tracked_frame.get_pose().detach()
                ref_pose = self.current_keyframe.get_pose().detach()
                rel_pose = torch.linalg.inv(ref_pose) @ tracked_pose
                self.frame_poses += [(len(self.keyframe_graph) -
                                      1, rel_pose.cpu())]

                if self.mesh_freq > 0 and (tracked_frame.index) % self.mesh_freq == 0:
                    if self.final_iter and len(self.keyframe_graph) > 20:
                        print(f"********** post-processing steps **********")
                        #self.num_iterations = 1
                        final_num_iter = len(self.keyframe_graph) + 1
                        progress_bar = tqdm(
                            range(0, final_num_iter), position=0)
                        progress_bar.set_description(" post-processing steps")
                        for iter in progress_bar:
                            #tracked_frame=self.keyframe_graph[iter//self.window_size]
                            self.do_mapping(share_data, tracked_frame=None,
                                            update_pose=False, update_decoder=False, selection_method='random')


                    self.logger.log_mesh(self.extract_mesh(res=self.mesh_res, clean_mesh=False),name=f"mesh_{tracked_frame.index:05d}.ply")
                    pose = self.get_updated_poses()
                    self.logger.log_numpy_data(np.asarray(pose), f"frame_poses_{tracked_frame.index:05d}")

                    if self.final_iter and len(self.keyframe_graph) > 20:
                        self.keyframe_graph = []
                        self.keyframe_graph += [self.current_keyframe]
                if self.save_data_freq > 0 and (tracked_frame.stamp + 1) % self.save_data_freq == 0:
                    self.save_debug_data(tracked_frame)
            elif share_data.stop_mapping:
                break
        print("******* extracting mesh without replay *******")
        self.logger.log_mesh(self.extract_mesh(res=self.mesh_res, clean_mesh=False), name="final_mesh_noreplay.ply")
        if self.final_iter:
            print(f"********** post-processing steps **********")
            #self.num_iterations = 1
            final_num_iter = len(self.keyframe_graph) + 1
            progress_bar = tqdm(
                range(0, final_num_iter), position=0)
            progress_bar.set_description(" post-processing steps")
            for iter in progress_bar:
                tracked_frame=self.keyframe_graph[iter//self.window_size]
                self.do_mapping(share_data, tracked_frame=None,
                                update_pose=False, update_decoder=False, selection_method='random')

        print("******* extracting final mesh *******")
        pose = self.get_updated_poses()
        self.logger.log_numpy_data(np.asarray(pose), "frame_poses")
        self.logger.log_mesh(self.extract_mesh(res=self.mesh_res, clean_mesh=False))
        print("******* mapping process died *******")

    def do_mapping(self, share_data, tracked_frame=None,
                   update_pose=True, update_decoder=True, selection_method = 'current'):
        self.profiler.tick("do_mapping")
        self.decoder.train()
        optimize_targets = self.select_optimize_targets(tracked_frame, selection_method=selection_method)
        torch.cuda.empty_cache()
        self.profiler.tick("bundle_adjust_frames")
        bundle_adjust_frames(
            optimize_targets,
            self.dynamic_embeddings,
            self.map_states,
            self.decoder,
            self.loss_criteria,
            self.voxel_size,
            self.step_size,
            self.n_rays * 2  if selection_method=='random' else self.n_rays,
            self.num_iterations,
            self.sdf_truncation,
            self.max_voxel_hit,
            self.max_distance,
            learning_rate=[self.learning_rate_emb, 
                           self.learning_rate_decorder, 
                           self.learning_rate_pose],
            update_pose=update_pose,
            update_decoder=update_decoder if tracked_frame == None or (tracked_frame.index -self.first_frame_id) < self.freeze_frame else False,
            profiler=self.profiler
        )
        self.profiler.tok("bundle_adjust_frames")
        # optimize_targets = [f.cpu() for f in optimize_targets]
        self.update_share_data(share_data)
        self.profiler.tok("do_mapping")
        # sleep(0.01)

    def select_optimize_targets(self, tracked_frame=None, selection_method='previous'):
        # TODO: better ways
        targets = []
        if selection_method == 'current':
            if tracked_frame == None:
                raise ValueError('select one track frame')
            else:
                return [tracked_frame]
        if len(self.keyframe_graph) <= self.window_size:
            targets = self.keyframe_graph[:]
        elif selection_method == 'random':
            targets = random.sample(self.keyframe_graph, self.window_size)
        elif selection_method == 'previous':
            targets = self.keyframe_graph[-self.window_size:]
        elif selection_method == 'overlap':
            raise NotImplementedError(
                f"seletion method {selection_method} unknown")

        if tracked_frame is not None and tracked_frame != self.current_keyframe:
            targets += [tracked_frame]
        return targets

    def update_share_data(self, share_data, frameid=None):
        share_data.decoder = deepcopy(self.decoder)
        tmp_states = {}
        for k, v in self.map_states.items():
            tmp_states[k] = v.detach().cpu()
        share_data.states = tmp_states
        # self.last_tracked_frame_id = frameid

    def remove_back_points(self, frame):
        rel_pose = frame.get_rel_pose()
        points = frame.get_points()
        points_norm = torch.norm(points, 2, -1)
        points_xy = points[:, :2]
        if rel_pose == None:
            x = 1
            y = 0
        else:
            x = rel_pose[0, 3]
            y = rel_pose[1, 3]
        rel_xy = torch.ones((1, 2))
        rel_xy[0, 0] = x
        rel_xy[0, 1] = y
        point_cos = torch.sum(-points_xy * rel_xy, dim=-1)/(
            torch.norm(points_xy, 2, -1)*(torch.norm(rel_xy, 2, -1)))
        remove_index = ((point_cos >= 0.7) & (points_norm > self.key_distance))
        new_points = frame.points[~remove_index]
        new_cos = frame.get_pointsCos()[~remove_index]
        return LidarFrame(frame.index, new_points, new_cos,
                          frame.pose, new_keyframe=True)

    def frame_maxdistance_change(self, frame, distance):
        # kf check
        valid_distance = distance + 0.5
        new_keyframe_rays_norm = frame.rays_norm.reshape(-1)
        new_keyframe_points = frame.points[new_keyframe_rays_norm <= valid_distance]
        new_keyframe_pointsCos = frame.get_pointsCos()[new_keyframe_rays_norm <= valid_distance]
        return LidarFrame(frame.index, new_keyframe_points, new_keyframe_pointsCos,
                          frame.pose, new_keyframe=True)

    def insert_keyframe(self, frame, valid_distance=-1):
        # kf check
        print("insert keyframe")
        valid_distance = self.key_distance + 0.01
        new_keyframe_rays_norm = frame.rays_norm.reshape(-1)
        mask = (torch.abs(frame.points[:, 0]) < valid_distance) & (torch.abs(frame.points[:, 1])
                                                                   < valid_distance) & (torch.abs(frame.points[:, 2]) < valid_distance)
        new_keyframe_points = frame.points[mask]
        new_keyframe_pointsCos = frame.get_pointsCos()[mask]
        new_keyframe = LidarFrame(frame.index, new_keyframe_points, new_keyframe_pointsCos,
                                  frame.pose, new_keyframe=True)
        if new_keyframe_points.shape[0] < 2*self.n_rays:
            raise ValueError('valid_distance too small')
        self.current_keyframe = new_keyframe
        self.keyframe_graph += [new_keyframe]
        # self.update_grid_features()

    def create_voxels(self, frame):
        points = frame.get_points().cuda()
        pose = frame.get_pose().cuda()
        print("frame id", frame.index+1)
        print("trans ", pose[:3, 3]-2000)
        points = points@pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
        voxels = torch.div(points, self.voxel_size, rounding_mode='floor')
        self.svo.insert(voxels.cpu().int())
        self.update_grid_features()

    @torch.enable_grad()
    def get_embeddings(self, points_idx):

        flatten_idx = points_idx.reshape(-1).long()
        valid_flatten_idx = flatten_idx[flatten_idx.ne(-1)]
        existence = F.embedding(valid_flatten_idx, self.voxel_id2embedding_id)
        torch_add_idx = existence.eq(-1).view(-1)
        torch_add = valid_flatten_idx[torch_add_idx]
        if torch_add.shape[0] == 0:
            return
        start_num = self.current_num_embeds
        end_num = start_num + torch_add.shape[0]
        embeddings_add = torch.zeros((end_num-start_num, self.embed_dim),
                                     dtype=torch.bfloat16)
        # torch.nn.init.normal_(embeddings_add, std=0.01)

        if self.dynamic_embeddings == None:
            embeddings = [embeddings_add]
        else:
            embeddings = [self.dynamic_embeddings.detach().cpu(), embeddings_add]
        embeddings = torch.cat(embeddings, dim=0)
        self.dynamic_embeddings = embeddings.cuda().requires_grad_()

        self.current_num_embeds = end_num
        self.voxel_id2embedding_id[torch_add] = torch.arange(start_num, end_num, dtype=torch.int).view(-1, 1)

    @torch.enable_grad()
    def update_grid_features(self):
        voxels, children, features = self.svo.get_centres_and_children()
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size
        children = torch.cat([children, voxels[:, -1:]], -1)

        centres = centres.float()
        children = children.int()

        map_states = {}
        map_states["voxel_vertex_idx"] = features
        centres.requires_grad_()
        map_states["voxel_center_xyz"] = centres
        map_states["voxel_structure"] = children
        self.profiler.tick("Creating embedding")
        self.get_embeddings(map_states["voxel_vertex_idx"])
        self.profiler.tok("Creating embedding")
        map_states["voxel_vertex_emb"] = self.dynamic_embeddings
        map_states["voxel_id2embedding_id"] = self.voxel_id2embedding_id

        self.map_states = map_states

    @torch.no_grad()
    def get_updated_poses(self, offset=-2000):
        for i in range(len(self.frame_poses)):
            ref_frame_ind, rel_pose = self.frame_poses[i]
            ref_frame = self.keyframe_graph[ref_frame_ind]
            ref_pose = ref_frame.get_pose().detach().cpu()
            pose = ref_pose @ rel_pose
            pose[:3, 3] += offset
            self.final_poses += [pose.detach().cpu().numpy()]
        self.frame_poses = []
        return self.final_poses

    @torch.no_grad()
    def extract_mesh(self, res=8, clean_mesh=False):
        sdf_network = self.decoder
        sdf_network.eval()

        voxels, _, features = self.svo.get_centres_and_children()
        index = features.eq(-1).any(-1)
        voxels = voxels[~index, :]
        features = features[~index, :]
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size

        encoder_states = {}
        encoder_states["voxel_vertex_idx"] = features
        encoder_states["voxel_center_xyz"] = centres
        self.profiler.tick("Creating embedding")
        self.get_embeddings(encoder_states["voxel_vertex_idx"])
        self.profiler.tok("Creating embedding")
        encoder_states["voxel_vertex_emb"] = self.dynamic_embeddings
        encoder_states["voxel_id2embedding_id"] = self.voxel_id2embedding_id

        frame_poses = self.get_updated_poses()
        mesh = self.mesher.create_mesh(
            self.decoder, encoder_states, self.voxel_size, voxels,
            frame_poses=None, depth_maps=None,
            clean_mseh=clean_mesh, require_color=False, offset=-2000, res=res)
        return mesh

    @torch.no_grad()
    def extract_voxels(self, offset=-10):
        voxels, _, features = self.svo.get_centres_and_children()
        index = features.eq(-1).any(-1)
        voxels = voxels[~index, :]
        features = features[~index, :]
        voxels = (voxels[:, :3] + voxels[:, -1:] / 2) * \
            self.voxel_size + offset
        # print(torch.max(features)-torch.count_nonzero(index))
        return voxels

    @torch.no_grad()
    def save_debug_data(self, tracked_frame, offset=-10):
        """
        save per-frame voxel, mesh and pose 
        """
        pose = tracked_frame.get_pose().detach().cpu().numpy()
        pose[:3, 3] += offset
        frame_poses = self.get_updated_poses()
        mesh = self.extract_mesh(res=8, clean_mesh=True)
        voxels = self.extract_voxels().detach().cpu().numpy()
        keyframe_poses = [p.get_pose().detach().cpu().numpy()
                          for p in self.keyframe_graph]

        for f in frame_poses:
            f[:3, 3] += offset
        for kf in keyframe_poses:
            kf[:3, 3] += offset

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        color = np.asarray(mesh.vertex_colors)

        self.logger.log_debug_data({
            "pose": pose,
            "updated_poses": frame_poses,
            "mesh": {"verts": verts, "faces": faces, "color": color},
            "voxels": voxels,
            "voxel_size": self.voxel_size,
            "keyframes": keyframe_poses,
            "is_keyframe": (tracked_frame == self.current_keyframe)
        }, tracked_frame.stamp)
