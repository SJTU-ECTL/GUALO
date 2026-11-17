import torch
import torch.nn.functional as F


def pairwise_loss(prediction, target, margin=0.05, temperature=0.5, elite=False):
    prediction, target = prediction.flatten(), target.detach().flatten()
    gap = target[:, None] - target[None, :]
    mask = torch.triu(torch.ones_like(gap, dtype=torch.bool), diagonal=1)
    mask &= gap.abs() > margin
    if not mask.any():
        return prediction.sum() * 0
    weights = torch.ones_like(gap)
    if elite:
        good = target <= torch.quantile(target, 0.3)
        weights += (good[:, None] | good[None, :]).float()
    loss = F.softplus(-gap.sign() * (prediction[:, None] - prediction[None, :]) / temperature)
    return (loss[mask] * weights[mask]).sum() / weights[mask].sum()


def uncertainty_loss(means, log_variances, target, rank_weight=0.1):
    mean = means.detach().mean(0)
    variance = means.detach().var(0, unbiased=False) + log_variances.exp().mean(0)
    variance = variance.clamp_min(1e-8)
    residual = (mean - target.flatten()).detach()
    nll = 0.5 * (variance.log() + residual.square() / variance).mean()
    rank = pairwise_loss(variance.log(), residual.abs())
    return nll + rank_weight * rank
