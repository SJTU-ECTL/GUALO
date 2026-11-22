import math
import numpy as np


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    _, inverse, counts = np.unique(values[order], return_inverse=True, return_counts=True)
    average = counts.cumsum() - (counts - 1) / 2
    result = np.empty(len(values), dtype=np.float64)
    result[order] = average[inverse]
    return result


def mape(predicted, actual):
    return {'mape_percent': float(np.mean(np.abs(predicted - actual) / actual) * 100)}


def ranking(predicted, actual):
    x, y = ranks(predicted), ranks(actual)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    rho = float(np.dot(x, y) / denominator) if denominator > 1e-12 else None
    k = max(1, math.ceil(len(actual) * .1))
    selected = set(np.argsort(predicted, kind='stable')[:k])
    best = set(np.argsort(actual, kind='stable')[:k])
    top10 = 100 * len(selected & best) / k if np.ptp(actual) > 0 else None
    return {'spearman': rho, 'top10_percent': top10}


def auc(labels, score):
    labels = np.asarray(labels, dtype=bool)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    return float((ranks(score)[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def uncertainty(predicted, actual, score, probability, scale, indices, fraction=.3, threshold=.5):
    indices = np.asarray(indices, dtype=int)
    if not len(indices) or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(actual):
        raise ValueError('Invalid UQ recipe indices')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid QoR scale')
    candidates = indices[predicted[indices] <= np.quantile(predicted[indices], fraction)]
    labels = (actual[candidates] - predicted[candidates]) / scale > threshold
    count = max(1, math.ceil(.2 * len(candidates)))
    selected = np.argsort(score[candidates], kind='stable')[-count:]
    positive = int(labels.sum())
    return {
        'auc': auc(labels, score[candidates]),
        'candidates': len(candidates),
        'underestimated': positive,
        'flagged': count,
        'caught': int(labels[selected].sum()),
        'precision_percent': float(labels[selected].mean() * 100),
        'recall_percent': float(labels[selected].sum() / positive * 100) if positive else None,
        'brier': float(np.mean((probability[candidates] - labels) ** 2)),
    }
