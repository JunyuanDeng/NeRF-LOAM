import torch


def sampling_without_replacement(logp, k):
    def gumbel_like(u):
        return -torch.log(-torch.log(torch.rand_like(u) + 1e-7) + 1e-7)

    scores = logp + gumbel_like(logp)
    return scores.topk(k, dim=-1)[1]


def sample_rays(mask, num_samples):
    B, H, W = mask.shape
    probs = mask / (mask.sum() + 1e-9)
    flatten_probs = probs.reshape(B, -1)
    sampled_index = sampling_without_replacement(
        torch.log(flatten_probs + 1e-9), num_samples)
    sampled_masks = (torch.zeros_like(
        flatten_probs).scatter_(-1, sampled_index, 1).reshape(B, H, W) > 0)
    return sampled_masks