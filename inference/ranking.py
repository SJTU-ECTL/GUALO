import math

import torch

if __package__:
    from .mape import arguments, evaluate
else:
    from mape import arguments, evaluate


def ranks(values):
    order = values.argsort(stable=True)
    _, inverse, counts = torch.unique_consecutive(values[order], return_inverse=True, return_counts=True)
    average = counts.cumsum(0).double() - (counts.double() - 1) / 2
    result = torch.empty_like(values, dtype=torch.float64)
    result[order] = average[inverse]
    return result


def ranking(predicted, actual):
    x, y = ranks(predicted), ranks(actual)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    rho = float((x @ y) / denominator) if denominator > 1e-12 else None
    k = max(1, math.ceil(len(actual) * .1))
    selected = set(predicted.argsort(stable=True)[:k].tolist())
    best = set(actual.argsort(stable=True)[:k].tolist())
    top10 = 100 * len(selected & best) / k if float(actual.max() - actual.min()) > 0 else None
    return dict(spearman=rho, top10_percent=top10)


if __name__ == '__main__':
    evaluate(arguments('ranking'), ranking)
