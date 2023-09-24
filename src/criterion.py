import torch
import torch.nn as nn
from torch.autograd import grad


class Criterion(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.eiko_weight = args.criteria["eiko_weight"]
        self.sdf_weight = args.criteria["sdf_weight"]
        self.fs_weight = args.criteria["fs_weight"]
        self.truncation = args.criteria["sdf_truncation"]
        self.max_dpeth = args.data_specs["max_depth"]

    def forward(self, outputs, obs, pointsCos, use_color_loss=True,
                use_depth_loss=True, compute_sdf_loss=True,
                weight_depth_loss=False, compute_eikonal_loss=False):

        points = obs
        loss = 0
        loss_dict = {}

        # pred_depth = outputs["depth"]
        pred_sdf = outputs["sdf"]
        z_vals = outputs["z_vals"]
        ray_mask = outputs["ray_mask"]
        valid_mask = outputs["valid_mask"]
        sampled_xyz = outputs["sampled_xyz"]
        gt_points = points[ray_mask]
        pointsCos = pointsCos[ray_mask]
        gt_distance = torch.norm(gt_points, 2, -1)

        gt_distance = gt_distance * pointsCos.view(-1)
        z_vals = z_vals * pointsCos.view(-1, 1)

        if compute_sdf_loss:
            fs_loss, sdf_loss, eikonal_loss = self.get_sdf_loss(
                z_vals, gt_distance, pred_sdf,
                truncation=self.truncation,
                loss_type='l2',
                valid_mask=valid_mask,
                compute_eikonal_loss=compute_eikonal_loss,
                points=sampled_xyz if compute_eikonal_loss else None
            )
            loss += self.fs_weight * fs_loss
            loss += self.sdf_weight * sdf_loss
            # loss += self.bs_weight * back_loss
            loss_dict["fs_loss"] = fs_loss.item()
            # loss_dict["bs_loss"] = back_loss.item()
            loss_dict["sdf_loss"] = sdf_loss.item()
            if compute_eikonal_loss:
                loss += self.eiko_weight * eikonal_loss
                loss_dict["eiko_loss"] = eikonal_loss.item()
        loss_dict["loss"] = loss.item()
        # print(loss_dict)
        return loss, loss_dict

    def compute_loss(self, x, y, mask=None, loss_type="l2"):
        if mask is None:
            mask = torch.ones_like(x).bool()
        if loss_type == "l1":
            return torch.mean(torch.abs(x - y)[mask])
        elif loss_type == "l2":
            return torch.mean(torch.square(x - y)[mask])

    def get_masks(self, z_vals, depth, epsilon):
        front_mask = torch.where(
            z_vals < (depth - epsilon),
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        back_mask = torch.where(
            z_vals > (depth + epsilon),
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        depth_mask = torch.where(
            (depth > 0.0) & (depth < self.max_dpeth), torch.ones_like(
                depth), torch.zeros_like(depth)
        )
        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

        num_fs_samples = torch.count_nonzero(front_mask).float()
        num_sdf_samples = torch.count_nonzero(sdf_mask).float()
        num_samples = num_sdf_samples + num_fs_samples
        fs_weight = 1.0 - num_fs_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        return front_mask, sdf_mask, fs_weight, sdf_weight

    def get_sdf_loss(self, z_vals, depth, predicted_sdf, truncation, valid_mask, loss_type="l2", compute_eikonal_loss=False, points=None):

        front_mask, sdf_mask, fs_weight, sdf_weight = self.get_masks(
            z_vals, depth.unsqueeze(-1).expand(*z_vals.shape), truncation
        )
        fs_loss = (self.compute_loss(predicted_sdf * front_mask * valid_mask, torch.ones_like(
            predicted_sdf) * front_mask, loss_type=loss_type,) * fs_weight)
        sdf_loss = (self.compute_loss((z_vals + predicted_sdf * truncation) * sdf_mask * valid_mask,
                    depth.unsqueeze(-1).expand(*z_vals.shape) * sdf_mask, loss_type=loss_type,) * sdf_weight)
        # back_loss = (self.compute_loss(predicted_sdf * back_mask, -torch.ones_like(
        #     predicted_sdf) * back_mask, loss_type=loss_type,) * back_weight)
        eikonal_loss = None
        if compute_eikonal_loss:
            sdf = (predicted_sdf*sdf_mask*truncation)
            sdf = sdf[valid_mask]
            d_points = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            sdf_grad = grad(outputs=sdf,
                            inputs=points,
                            grad_outputs=d_points,
                            retain_graph=True,
                            only_inputs=True)[0]
            eikonal_loss = self.compute_loss(sdf_grad[0].norm(2, -1), 1.0, loss_type=loss_type,)

        return fs_loss, sdf_loss, eikonal_loss
