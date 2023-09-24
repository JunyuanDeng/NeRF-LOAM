import os
import os.path as osp
import pickle
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import yaml


class BasicLogger:
    def __init__(self, args) -> None:
        self.args = args
        self.log_dir = osp.join(
            args.log_dir, args.exp_name, self.get_random_time_str())
        self.img_dir = osp.join(self.log_dir, "imgs")
        self.mesh_dir = osp.join(self.log_dir, "mesh")
        self.ckpt_dir = osp.join(self.log_dir, "ckpt")
        self.backup_dir = osp.join(self.log_dir, "bak")
        self.misc_dir = osp.join(self.log_dir, "misc")

        os.makedirs(self.img_dir)
        os.makedirs(self.ckpt_dir)
        os.makedirs(self.mesh_dir)
        os.makedirs(self.misc_dir)
        os.makedirs(self.backup_dir)

        self.log_config(args)

    def get_random_time_str(self):
        return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")

    def log_ckpt(self, mapper):
        print("******* saving *******")
        decoder_state = {f: v.cpu()
                         for f, v in mapper.decoder.state_dict().items()}
        map_state = {f: v.cpu() for f, v in mapper.map_states.items()}
        embeddings = mapper.dynamic_embeddings.cpu()
        svo = mapper.svo
        torch.save({
            "decoder_state": decoder_state,
            # "map_state": map_state,
            "embeddings": embeddings,
            "svo": svo
        },
            os.path.join(self.ckpt_dir, "final_ckpt.pth"))
        print("******* finish saving *******")

    def log_config(self, config):
        out_path = osp.join(self.backup_dir, "config.yaml")
        yaml.dump(config, open(out_path, 'w'))

    def log_mesh(self, mesh, name="final_mesh.ply"):
        out_path = osp.join(self.mesh_dir, name)
        o3d.io.write_triangle_mesh(out_path, mesh)

    def log_point_cloud(self, pcd, name="final_points.ply"):
        out_path = osp.join(self.mesh_dir, name)
        o3d.io.write_point_cloud(out_path, pcd)

    def log_numpy_data(self, data, name, ind=None):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if ind is not None:
            np.save(osp.join(self.misc_dir, "{}-{:05d}.npy".format(name, ind)), data)
        else:
            np.save(osp.join(self.misc_dir, f"{name}.npy"), data)
            self.npy2txt(osp.join(self.misc_dir, f"{name}.npy"), osp.join(self.misc_dir, f"{name}.txt"))

    def log_debug_data(self, data, idx):
        with open(os.path.join(self.misc_dir, f"scene_data_{idx}.pkl"), 'wb') as f:
            pickle.dump(data, f)

    def log_raw_image(self, ind, rgb, depth):
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        rgb = cv2.cvtColor(rgb*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join(self.img_dir, "{:05d}.jpg".format(
            ind)), (rgb).astype(np.uint8))
        cv2.imwrite(osp.join(self.img_dir, "{:05d}.png".format(
            ind)), (depth*5000).astype(np.uint16))

    def log_images(self, ind, gt_rgb, gt_depth, rgb, depth):
        gt_depth_np = gt_depth.detach().cpu().numpy()
        gt_color_np = gt_rgb.detach().cpu().numpy()
        depth_np = depth.squeeze().detach().cpu().numpy()
        color_np = rgb.detach().cpu().numpy()

        h, w = depth_np.shape
        gt_depth_np = cv2.resize(
            gt_depth_np, (w, h), interpolation=cv2.INTER_NEAREST)
        gt_color_np = cv2.resize(
            gt_color_np, (w, h), interpolation=cv2.INTER_AREA)

        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        color_residual = np.abs(gt_color_np - color_np)
        color_residual[gt_depth_np == 0.0] = 0.0

        fig, axs = plt.subplots(2, 3)
        fig.tight_layout()
        max_depth = np.max(gt_depth_np)
        axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                         vmin=0, vmax=max_depth)
        axs[0, 0].set_title('Input Depth')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="plasma",
                         vmin=0, vmax=max_depth)
        axs[0, 1].set_title('Generated Depth')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma",
                         vmin=0, vmax=max_depth)
        axs[0, 2].set_title('Depth Residual')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Generated RGB')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 2].imshow(color_residual, cmap="plasma")
        axs[1, 2].set_title('RGB Residual')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(osp.join(self.img_dir, "{:05d}.jpg".format(
            ind)), bbox_inches='tight', pad_inches=0.2)
        plt.clf()
        plt.close()

    def npy2txt(self, input_path, output_path):
        poses = np.load(input_path)
        with open(output_path, mode='w') as w:
            shape = poses.shape
            print(shape)
            for i in range(shape[0]):
                one_pose = str()
                for j in range(shape[1]):
                    if j == (shape[1]-1):
                        continue
                    for k in range(shape[2]):
                        if j == (shape[1]-2) and k == (shape[1]-1):
                            one_pose += (str(poses[i][j][k])+"\n")
                        else:
                            one_pose += (str(poses[i][j][k])+" ")
                w.write(one_pose)
