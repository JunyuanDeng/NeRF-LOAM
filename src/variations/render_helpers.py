from copy import deepcopy
import torch
import torch.nn.functional as F

from .voxel_helpers import ray_intersect, ray_sample
from torch.autograd import grad


def ray(ray_start, ray_dir, depths):
    return ray_start + ray_dir * depths


def fill_in(shape, mask, input, initial=1.0):
    if isinstance(initial, torch.Tensor):
        output = initial.expand(*shape)
    else:
        output = input.new_ones(*shape) * initial
    return output.masked_scatter(mask.unsqueeze(-1).expand(*shape), input)


def masked_scatter(mask, x):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_zeros(B, K).masked_scatter(mask, x)
    return x.new_zeros(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


def masked_scatter_ones(mask, x):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_ones(B, K).masked_scatter(mask, x)
    return x.new_ones(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


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
    offset = (torch.cat([
        ox.reshape(-1, 1),
        oy.reshape(-1, 1),
        oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
    if not offset_only:
        return (
            point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel)
    return offset.type_as(point_xyz) * quarter_voxel


@torch.enable_grad()
def get_embeddings(sampled_xyz, point_xyz, point_feats, voxel_size):
    # tri-linear interpolation
    p = ((sampled_xyz - point_xyz) / voxel_size + 0.5).unsqueeze(1)
    q = offset_points(p, 0.5, offset_only=True).unsqueeze(0) + 0.5
    feats = trilinear_interp(p, q, point_feats).float()
    # if self.args.local_coord:
    # feats = torch.cat([(p-.5).squeeze(1).float(), feats], dim=-1)
    return feats


@torch.enable_grad()
def get_features(samples, map_states, voxel_size):
    # encoder states
    point_idx = map_states["voxel_vertex_idx"].cuda()
    point_xyz = map_states["voxel_center_xyz"].cuda()
    values = map_states["voxel_vertex_emb"]
    point_id2embedid = map_states["voxel_id2embedding_id"]
    # ray point samples
    sampled_idx = samples["sampled_point_voxel_idx"].long()
    sampled_xyz = samples["sampled_point_xyz"]
    sampled_dis = samples["sampled_point_distance"]

    point_xyz = F.embedding(sampled_idx, point_xyz).requires_grad_()
    selected_points_idx = F.embedding(sampled_idx, point_idx)
    flatten_selected_points_idx = selected_points_idx.view(-1)
    embed_idx = F.embedding(flatten_selected_points_idx.cpu(), point_id2embedid).squeeze(-1)
    point_feats = F.embedding(embed_idx.cuda(), values).view(point_xyz.size(0), -1)

    feats = get_embeddings(sampled_xyz, point_xyz, point_feats, voxel_size)
    inputs = {"xyz": point_xyz, "dists": sampled_dis, "emb": feats.cuda()}
    return inputs


@torch.no_grad()
def get_scores(sdf_network, map_states, voxel_size, bits=8):
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz"]
    values = map_states["voxel_vertex_emb"]
    point_id2embedid = map_states["voxel_id2embedding_id"]

    chunk_size = 10000
    res = bits  # -1

    @torch.no_grad()
    def get_scores_once(feats, points, values, point_id2embedid):
        torch.cuda.empty_cache()
        # sample points inside voxels
        start = -0.5
        end = 0.5  # - 1./bits

        x = y = z = torch.linspace(start, end, res)
        # z = torch.linspace(1, 1, res)
        xx, yy, zz = torch.meshgrid(x, y, z)
        sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float().cuda()

        sampled_xyz *= voxel_size
        sampled_xyz = sampled_xyz.reshape(1, -1, 3) + points.unsqueeze(1)

        sampled_idx = torch.arange(points.size(0), device=points.device)
        sampled_idx = sampled_idx[:, None].expand(*sampled_xyz.size()[:2])
        sampled_idx = sampled_idx.reshape(-1)
        sampled_xyz = sampled_xyz.reshape(-1, 3)

        if sampled_xyz.shape[0] == 0:
            return

        field_inputs = get_features(
            {
                "sampled_point_xyz": sampled_xyz,
                "sampled_point_voxel_idx": sampled_idx,
                "sampled_point_ray_direction": None,
                "sampled_point_distance": None,
            },
            {
                "voxel_vertex_idx": feats,
                "voxel_center_xyz": points,
                "voxel_vertex_emb": values,
                "voxel_id2embedding_id": point_id2embedid
            },
            voxel_size
        )
        field_inputs = field_inputs["emb"]

        # evaluation with density
        sdf_values = sdf_network.get_values(field_inputs.float().cuda())
        return sdf_values.reshape(-1, res ** 3, 1).detach().cpu()

    return torch.cat([
        get_scores_once(feats[i: i + chunk_size],
                        points[i: i + chunk_size].cuda(), values, point_id2embedid)
        for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 1)


@torch.no_grad()
def eval_points(sdf_network, map_states, sampled_xyz, sampled_idx, voxel_size):
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz"]
    values = map_states["voxel_vertex_emb"]

    sampled_idx = sampled_idx.reshape(-1)
    sampled_xyz = sampled_xyz.reshape(-1, 3)

    if sampled_xyz.shape[0] == 0:
        return

    field_inputs = get_features(
        {
            "sampled_point_xyz": sampled_xyz,
            "sampled_point_voxel_idx": sampled_idx,
            "sampled_point_ray_direction": None,
            "sampled_point_distance": None,
        },
        {
            "voxel_vertex_idx": feats,
            "voxel_center_xyz": points,
            "voxel_vertex_emb": values,
        },
        voxel_size
    )

    # evaluation with density
    sdf_values = sdf_network.get_values(field_inputs['emb'].float().cuda())
    return sdf_values.reshape(-1, 4)[:, :3].detach().cpu()

    # return torch.cat([
    #     get_scores_once(feats[i: i + chunk_size],
    #                     points[i: i + chunk_size], values)
    #     for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 4)


def render_rays(
        rays_o,
        rays_d,
        map_states,
        sdf_network,
        step_size,
        voxel_size,
        truncation,
        max_voxel_hit,
        max_distance,
        chunk_size=10000,
        profiler=None,
        return_raw=False
):
    torch.cuda.empty_cache()
    centres = map_states["voxel_center_xyz"].cuda()
    childrens = map_states["voxel_structure"].cuda()

    if profiler is not None:
        profiler.tick("ray_intersect")
    # print("Center", rays_o[0][0])
    intersections, hits = ray_intersect(
        rays_o, rays_d, centres,
        childrens, voxel_size, max_voxel_hit, max_distance)
    if profiler is not None:
        profiler.tok("ray_intersect")
    if hits.sum() <= 0:
        return

    ray_mask = hits.view(1, -1)

    intersections = {
        name: outs[ray_mask].reshape(-1, outs.size(-1))
        for name, outs in intersections.items()
    }

    rays_o = rays_o[ray_mask].reshape(-1, 3)
    rays_d = rays_d[ray_mask].reshape(-1, 3)

    if profiler is not None:
        profiler.tick("ray_sample")
    samples = ray_sample(intersections, step_size=step_size)
    if samples == None:
        return
    if profiler is not None:
        profiler.tok("ray_sample")

    sampled_depth = samples['sampled_point_depth']
    sampled_idx = samples['sampled_point_voxel_idx'].long()

    # only compute when the ray hits

    sample_mask = sampled_idx.ne(-1)
    if sample_mask.sum() == 0:  # miss everything skip
        return None, 0

    sampled_xyz = ray(rays_o.unsqueeze(
        1), rays_d.unsqueeze(1), sampled_depth.unsqueeze(2))
    sampled_dir = rays_d.unsqueeze(1).expand(
        *sampled_depth.size(), rays_d.size()[-1])
    sampled_dir = sampled_dir / \
        (torch.norm(sampled_dir, 2, -1, keepdim=True) + 1e-8)
    samples['sampled_point_xyz'] = sampled_xyz
    samples['sampled_point_ray_direction'] = sampled_dir
    # apply mask
    samples_valid = {name: s[sample_mask] for name, s in samples.items()}
    num_points = samples_valid['sampled_point_depth'].shape[0]
    field_outputs = []

    if chunk_size < 0:
        chunk_size = num_points
    final_xyz = []
    xyz = 0
    for i in range(0, num_points, chunk_size):
        torch.cuda.empty_cache()
        chunk_samples = {name: s[i:i+chunk_size]
                         for name, s in samples_valid.items()}

        # get encoder features as inputs
        if profiler is not None:
            profiler.tick("get_features")

        chunk_inputs = get_features(chunk_samples, map_states, voxel_size)
        xyz = chunk_inputs["xyz"]
        if profiler is not None:
            profiler.tok("get_features")
        # add coordinate information
        chunk_inputs = chunk_inputs["emb"]
        # chunk_inputs = torch.cat([chunk_inputs, rays_o[0,:].expand(chunk_inputs.shape[0], 3)],\
        #                dim=-1)

        # forward implicit fields
        if profiler is not None:
            profiler.tick("render_core")

        chunk_outputs = sdf_network(chunk_inputs)
        if profiler is not None:
            profiler.tok("render_core")
        final_xyz.append(xyz)
        field_outputs.append(chunk_outputs)

    field_outputs = {name: torch.cat(
        [r[name] for r in field_outputs], dim=0) for name in field_outputs[0]}
    final_xyz = torch.cat(final_xyz, 0)

    outputs = field_outputs['sdf']

    outputs = {'sample_mask': sample_mask}

    sdf = masked_scatter_ones(sample_mask, field_outputs['sdf']).squeeze(-1)

    # colour = torch.sigmoid(colour)
    sample_mask = outputs['sample_mask']

    valid_mask = torch.where(
        sample_mask, torch.ones_like(
            sample_mask), torch.zeros_like(sample_mask)
    )
    return {
        "z_vals": samples["sampled_point_depth"],
        "sdf": sdf,
        "ray_mask": ray_mask,
        "valid_mask": valid_mask,
        "sampled_xyz": xyz,
    }


def bundle_adjust_frames(
    keyframe_graph,
    embeddings,
    map_states,
    sdf_network,
    loss_criteria,
    voxel_size,
    step_size,
    N_rays=512,
    num_iterations=10,
    truncation=0.1,
    max_voxel_hit=10,
    max_distance=10,
    learning_rate=[1e-2, 1e-2, 5e-3],
    update_pose=True,
    update_decoder=True,
    profiler=None
):
    if profiler is not None:
        profiler.tick("mapping_add_optim")
    optimize_params = [{'params': embeddings, 'lr': learning_rate[0]}]
    if update_decoder:
        optimize_params += [{'params': sdf_network.parameters(),
                             'lr': learning_rate[1]}]

    for keyframe in keyframe_graph:
        if keyframe.index != 0 and update_pose:
            keyframe.pose.requires_grad_(True)
            optimize_params += [{
                'params': keyframe.pose.parameters(), 'lr': learning_rate[2]
            }]

    optim = torch.optim.Adam(optimize_params)
    if profiler is not None:
        profiler.tok("mapping_add_optim")
    for iter in range(num_iterations):
        torch.cuda.empty_cache()
        rays_o = []
        rays_d = []
        points_samples = []
        pointsCos_samples = []
        if iter == 0 and profiler is not None:
            profiler.tick("mapping sample_rays")
        for frame in keyframe_graph:
            torch.cuda.empty_cache()
            pose = frame.get_pose().cuda()
            frame.sample_rays(N_rays)

            sample_mask = frame.sample_mask.cuda()
            sampled_rays_d = frame.rays_d[sample_mask].cuda()
            R = pose[: 3, : 3].transpose(-1, -2)
            sampled_rays_d = sampled_rays_d@R
            sampled_rays_o = pose[: 3, 3].reshape(1, -1).expand_as(sampled_rays_d)

            rays_d += [sampled_rays_d]
            rays_o += [sampled_rays_o]
            points_samples += [frame.points.unsqueeze(1).cuda()[sample_mask]]
            pointsCos_samples += [frame.pointsCos.unsqueeze(1).cuda()[sample_mask]]

        rays_d = torch.cat(rays_d, dim=0).unsqueeze(0)
        rays_o = torch.cat(rays_o, dim=0).unsqueeze(0)
        points_samples = torch.cat(points_samples, dim=0).unsqueeze(0)
        pointsCos_samples = torch.cat(pointsCos_samples, dim=0).unsqueeze(0)
        if iter == 0 and profiler is not None:
            profiler.tok("mapping sample_rays")
        if iter == 0 and profiler is not None:
            profiler.tick("mapping rendering")
        final_outputs = render_rays(
            rays_o,
            rays_d,
            map_states,
            sdf_network,
            step_size,
            voxel_size,
            truncation,
            max_voxel_hit,
            max_distance,
            chunk_size=-1,
            profiler=profiler if iter == 0 else None
        )
        if final_outputs == None:
            print("Encouter a bug while Mapping, currently not be fixed, Continue!!")
            hit_mask = None
            continue
        if iter == 0 and profiler is not None:
            profiler.tok("mapping rendering")
        if iter == 0 and profiler is not None:
            profiler.tick("mapping back proj")
        torch.cuda.empty_cache()
        loss, _ = loss_criteria(
            final_outputs, points_samples, pointsCos_samples)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if iter == 0 and profiler is not None:
            profiler.tok("mapping back proj")


def track_frame(
    frame_pose,
    curr_frame,
    map_states,
    sdf_network,
    loss_criteria,
    voxel_size,
    N_rays=512,
    step_size=0.05,
    num_iterations=10,
    truncation=0.1,
    learning_rate=1e-3,
    max_voxel_hit=10,
    max_distance=10,
    profiler=None,
    depth_variance=False
):
    torch.cuda.empty_cache()
    init_pose = deepcopy(frame_pose).cuda()
    init_pose.requires_grad_(True)
    optim = torch.optim.Adam(init_pose.parameters(),
                             lr=learning_rate*2 if curr_frame.index < 2
                             else learning_rate/3)

    for iter in range(num_iterations):
        torch.cuda.empty_cache()
        if iter == 0 and profiler is not None:
            profiler.tick("track sample_rays")
        curr_frame.sample_rays(N_rays, track=True)
        if iter == 0 and profiler is not None:
            profiler.tok("track sample_rays")

        sample_mask = curr_frame.sample_mask
        ray_dirs = curr_frame.rays_d[sample_mask].unsqueeze(0).cuda()
        # print(curr_frame.rays_d[sample_mask][:,2].max())
        points_samples = curr_frame.points.unsqueeze(1).cuda()[sample_mask]
        pointsCos_samples = curr_frame.pointsCos.unsqueeze(1).cuda()[sample_mask]
        # rgb = curr_frame.rgb[sample_mask].cuda()
        # depth = curr_frame.depth[sample_mask].cuda()

        ray_dirs_iter = ray_dirs.squeeze(
            0) @ init_pose.rotation().transpose(-1, -2)
        ray_dirs_iter = ray_dirs_iter.unsqueeze(0)
        ray_start_iter = init_pose.translation().reshape(
            1, 1, -1).expand_as(ray_dirs_iter).cuda().contiguous()

        if iter == 0 and profiler is not None:
            profiler.tick("track render_rays")
        final_outputs = render_rays(
            ray_start_iter,
            ray_dirs_iter,
            map_states,
            sdf_network,
            step_size,
            voxel_size,
            truncation,
            max_voxel_hit,
            max_distance,
            chunk_size=-2,
            profiler=profiler if iter == 0 else None
        )
        if final_outputs == None:
            print("Encouter a bug while Tracking, currently not be fixed, Restarting!!")
            hit_mask = None
            break
        # if final_outputs == None:
        #    continue
        torch.cuda.empty_cache()
        if iter == 0 and profiler is not None:
            profiler.tok("track render_rays")

        hit_mask = final_outputs["ray_mask"].view(N_rays)
        final_outputs["ray_mask"] = hit_mask

        if iter == 0 and profiler is not None:
            profiler.tick("track loss_criteria")
        loss, _ = loss_criteria(
            final_outputs, points_samples, pointsCos_samples, weight_depth_loss=depth_variance)
        if iter == 0 and profiler is not None:
            profiler.tok("track loss_criteria")
        if iter == 0 and profiler is not None:
            profiler.tick("track backward step")
        optim.zero_grad()
        loss.backward()
        optim.step()
        if iter == 0 and profiler is not None:
            profiler.tok("track backward step")

    return init_pose, hit_mask
