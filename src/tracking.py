import torch
import numpy as np
from tqdm import tqdm

from criterion import Criterion
from lidarFrame import LidarFrame
from utils.import_util import get_property
from utils.profile_util import Profiler
from variations.render_helpers import fill_in, render_rays, track_frame
from se3pose import OptimizablePose
from time import sleep
from copy import deepcopy


class Tracking:
    def __init__(self, args, data_stream, logger):
        self.args = args
        self.last_frame_id = 0
        self.last_frame = None

        self.data_stream = data_stream
        self.logger = logger
        self.loss_criteria = Criterion(args)

        self.voxel_size = args.mapper_specs["voxel_size"]
        self.N_rays = args.tracker_specs["N_rays"]
        self.num_iterations = args.tracker_specs["num_iterations"]
        self.sdf_truncation = args.criteria["sdf_truncation"]
        self.learning_rate = args.tracker_specs["learning_rate"]
        self.start_frame = args.tracker_specs["start_frame"]
        self.end_frame = args.tracker_specs["end_frame"]
        self.step_size = args.tracker_specs["step_size"]
        # self.keyframe_freq = args.tracker_specs["keyframe_freq"]
        self.max_voxel_hit = args.tracker_specs["max_voxel_hit"]
        self.max_distance = args.data_specs["max_depth"]
        self.step_size = self.step_size * self.voxel_size
        self.read_offset = args.tracker_specs["read_offset"]
        self.mesh_freq = args.debug_args["mesh_freq"]
        if self.end_frame <= 0:
            self.end_frame = len(self.data_stream)-1

        # sanity check on the lower/upper bounds
        self.start_frame = min(self.start_frame, len(self.data_stream))
        self.end_frame = min(self.end_frame, len(self.data_stream))
        self.rel_pose = None
        # profiler
        verbose = get_property(args.debug_args, "verbose", False)
        self.profiler = Profiler(verbose=verbose)
        self.profiler.enable()

    def process_first_frame(self, kf_buffer):
        init_pose = self.data_stream.get_init_pose(self.start_frame)
        index, points, pointcos, _ = self.data_stream[self.start_frame]
        first_frame = LidarFrame(index, points, pointcos, init_pose)
        first_frame.pose.requires_grad_(False)
        first_frame.points.requires_grad_(False)

        print("******* initializing first_frame:", first_frame.index)
        kf_buffer.put(first_frame, block=True)
        self.last_frame = first_frame
        self.start_frame += 1

    def process_restart_frame(self, kf_buffer, frame_id, restart_pose):
        restart_pose = OptimizablePose(restart_pose)
        index, points, pointcos, _ = self.data_stream[frame_id]
        first_frame = LidarFrame(index, points, pointcos, restart_pose, new_keyframe=True)
        first_frame.pose.requires_grad_(False)
        first_frame.points.requires_grad_(False)

        print("******* initializing restart_frame:", first_frame.index)
        kf_buffer.put(first_frame, block=True)
        self.last_frame = first_frame

    def spin(self, share_data, kf_buffer):
        print("******* tracking process started! *******")
        progress_bar = tqdm(
            range(self.start_frame, self.end_frame+1), position=0)
        progress_bar.set_description("tracking frame")
        for frame_id in progress_bar:
            if frame_id % self.read_offset != 0:
                continue
            if share_data.stop_tracking:
                break

            data_in = self.data_stream[frame_id]

            current_frame = LidarFrame(*data_in)
            if isinstance(data_in[3], np.ndarray):
                self.last_frame = current_frame
                self.check_keyframe(current_frame, kf_buffer)

            else:
                self.do_tracking(share_data, current_frame, kf_buffer)

            if frame_id % self.mesh_freq == 0:
                share_data.wait_restart = True
                while share_data.wait_restart:
                    sleep(0.1)
                self.process_restart_frame(kf_buffer, frame_id, share_data.restart_pose)
                self.rel_pose = None

        share_data.stop_mapping = True
        print("******* tracking process died *******")

        sleep(60)
        while not kf_buffer.empty():
            sleep(60)

    def check_keyframe(self, check_frame, kf_buffer):
        try:
            kf_buffer.put(check_frame, block=True)
        except:
            pass

    def do_tracking(self, share_data, current_frame, kf_buffer):

        self.profiler.tick("before track1111")
        decoder = share_data.decoder.cuda()
        self.profiler.tok("before track1111")

        self.profiler.tick("before track2222")
        map_states = share_data.states
        map_states["voxel_vertex_emb"] = map_states["voxel_vertex_emb"].cuda()
        self.profiler.tok("before track2222")

        constant_move_pose = self.last_frame.get_pose().detach()
        input_pose = deepcopy(self.last_frame.pose)
        input_pose.requires_grad_(False)
        if self.rel_pose != None:
            print("before")
            print(input_pose.data)
            # print(constant_move_pose)
            constant_move_pose[:3, 3] = (constant_move_pose @ (self.rel_pose))[:3, 3]
            input_pose.data[:3] = constant_move_pose[:3, 3].T
            #print("after")
            #print(input_pose.data)
            # print(constant_move_pose)
        torch.cuda.empty_cache()
        self.profiler.tick("track frame")

        frame_pose, hit_mask = track_frame(
            input_pose,
            current_frame,
            map_states,
            decoder,
            self.loss_criteria,
            self.voxel_size,
            self.N_rays,
            self.step_size,
            self.num_iterations if self.rel_pose != None else self.num_iterations*5,
            self.sdf_truncation,
            self.learning_rate if self.rel_pose != None else self.learning_rate,
            self.max_voxel_hit,
            self.max_distance,
            profiler=self.profiler,
            depth_variance=True
        )
        self.profiler.tok("track frame")
        if hit_mask == None:
            current_frame.pose = OptimizablePose.from_matrix(constant_move_pose)
        else:
            current_frame.pose = frame_pose
            current_frame.hit_ratio = hit_mask.sum() / self.N_rays

        self.rel_pose = torch.linalg.inv(self.last_frame.get_pose().detach()) @ current_frame.get_pose().detach()
        current_frame.set_rel_pose(self.rel_pose)
        self.last_frame = current_frame

        self.profiler.tick("transport frame")
        self.check_keyframe(current_frame, kf_buffer)
        self.profiler.tok("transport frame")
