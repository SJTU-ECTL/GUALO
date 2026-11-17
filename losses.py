import torch
from torch.nn import functional as F


def nll_loss(mu, log_var, target):
    mu, log_var, target = mu.flatten(), log_var.flatten(), target.flatten()
    return (0.5 * log_var + 0.5 * (target - mu).square() * (-log_var).exp()).mean()


def select_recipes(scores, top_k=10, margin=0.05, negative_ratio=3):

    scores = scores.detach().flatten()
    if top_k < 1 or negative_ratio < 1 or not 0 <= margin <= 1:
        raise ValueError('Invalid contrastive selection settings')
    empty = torch.empty(0, dtype=torch.long, device=scores.device)
    if len(scores) < 4 or float(scores.max() - scores.min()) < 1e-8:
        return empty, empty
    order = torch.argsort(scores, descending=True, stable=True)
    positive = order[:min(top_k, len(scores) // 4)]
    threshold = scores.max() - margin * (scores.max() - scores.min())
    eligible = scores <= threshold
    eligible[positive] = False
    negative = order[eligible[order]][-negative_ratio * len(positive):]
    return positive, negative


def contrastive_losses(graph_proj, recipe_proj, scores, temperature=0.1,
                       top_k=10, margin=0.05, negative_ratio=3):

    if graph_proj.ndim != 2 or graph_proj.shape[0] != 1:
        raise ValueError('Each contrastive batch must contain exactly one circuit anchor')
    if recipe_proj.ndim != 2 or len(recipe_proj) != scores.numel():
        raise ValueError('Recipe projections and scores must have matching lengths')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    positive, negative = select_recipes(scores, top_k, margin, negative_ratio)
    zero = (graph_proj.sum() + recipe_proj.sum()) * 0
    if not len(positive) or not len(negative):
        return zero, zero
    graph = F.normalize(graph_proj, dim=-1)
    recipes = F.normalize(recipe_proj, dim=-1)
    selected = torch.cat((positive, negative))
    similarities = graph @ recipes.T / temperature
    cross = (similarities[:, selected].logsumexp(1)
             - similarities[:, positive].logsumexp(1)).mean()
    if len(positive) < 2:
        return zero, cross
    similarities = recipes[positive] @ recipes.T / temperature
    mask = torch.arange(len(recipes), device=recipes.device)[None, :] == positive[:, None]
    similarities = similarities.masked_fill(mask, -torch.inf)
    recipe = (similarities[:, selected].logsumexp(1)
              - similarities[:, positive].logsumexp(1)).mean()
    return recipe, cross


def joint_loss(mu, log_var, target, graph_proj, recipe_proj, *,
               lambda_recipe=0.5, lambda_cross=0.3, temperature=0.1,
               top_k=10, margin=0.05, negative_ratio=3):

    qor = nll_loss(mu, log_var, target)
    recipe, cross = contrastive_losses(graph_proj, recipe_proj, -target,
                                      temperature, top_k, margin, negative_ratio)
    return qor + lambda_recipe * recipe + lambda_cross * cross, qor, recipe, cross
