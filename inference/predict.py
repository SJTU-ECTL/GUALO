import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.engine import GUALOEngine


class GUALOPredictor:
    def __init__(self, checkpoint=ROOT / 'GUALO_test_demo.pt', device=None):
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        package = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if package.get('format') != 'gualo-inference-v1':
            raise ValueError('Unsupported GUALO model package')
        self.engines = {}
        for target in ('area', 'delay'):
            branch = package['targets'][target]
            if branch['risk']['target'] != target:
                raise ValueError('Model target mismatch')
            self.engines[target] = GUALOEngine(branch, device)

    def scale(self, design, target):
        return self.engines[target].risk.settings['designs'][design]['normalization'][1]

    def predict(self, design, data):
        tokens = data['opt_seq']
        if tokens.ndim != 2 or not len(tokens):
            raise ValueError('Expected a nonempty recipe matrix')
        valid = tokens.ne(0)
        padding = torch.arange(tokens.shape[1])[None, :] < valid.sum(1)[:, None]
        if not valid.any(1).all() or not torch.equal(valid, padding) or not ((tokens >= 0) & (tokens < 14) & (tokens == tokens.long())).all():
            raise ValueError('Invalid recipe tokens or padding')
        result = {}
        for target, engine in self.engines.items():
            output = engine.predict(design, data)
            result[target] = output['mean']
            result[target + '_risk'] = output['optimistic_probability']
            result[target + '_risk_score'] = output['raw_probability']
            result[target + '_cost_score'] = output['cost_score']
            result[target + '_risk_penalty'] = output['optimism_penalty']
            result[target + '_scale'] = np.asarray(self.scale(design, target))
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT / 'GUALO_test_demo.pt')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('predictions.npz'))
    parser.add_argument('--device')
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = torch.load(args.dataset, weights_only=True, map_location='cpu')
    design = args.dataset.stem.removeprefix('OpenCores__')
    result = GUALOPredictor(args.model, args.device).predict(design, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(args.output)
