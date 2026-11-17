import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_partition(tokens, seed=42):
    value = int.from_bytes(hashlib.sha256(bytes(tokens) + f':{seed}'.encode()).digest()[:8], 'big') % 1000
    return 'train' if value < 800 else 'validation' if value < 900 else 'test'


class CircuitDataset:

    def __init__(self, directory, target, seed=42):
        root = Path(directory).resolve()
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('format_version') != 'gualo-tensors-v1':
            raise ValueError('Expected a gualo-tensors-v1 manifest')
        self.items, records, names = [], [], set()
        for spec in manifest['circuits']:
            name = spec['id']
            if not isinstance(name, str) or not name or name in names:
                raise ValueError('Circuit ids must be unique nonempty strings')
            names.add(name)
            path = (root / spec['file']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Circuit files must be inside the dataset directory')
            blob = torch.load(path, map_location='cpu', weights_only=True)
            x, edges, seq, raw = (blob[k] for k in ('x', 'edge_index', 'opt_seq', target))
            if x.ndim != 2 or x.shape[1] != 5 or len(x) < 2 or x.is_floating_point():
                raise ValueError(f'{name}: x must be integer one-hot [nodes>=2,5]')
            if not ((x == 0) | (x == 1)).all() or not (x.sum(1) == 1).all():
                raise ValueError(f'{name}: invalid node one-hot features')
            if edges.ndim != 2 or edges.shape[0] != 2 or edges.is_floating_point():
                raise ValueError(f'{name}: invalid edge_index')
            if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= len(x)):
                raise ValueError(f'{name}: edge endpoint out of range')
            if seq.ndim != 2 or seq.shape[1] == 0 or seq.is_floating_point():
                raise ValueError(f'{name}: opt_seq must be integer [recipes,length]')
            valid = seq.ne(0)
            if not valid.any(1).all() or not ((seq >= 0) & (seq <= 13)).all():
                raise ValueError(f'{name}: nonempty recipes require tokens 1..13 and padding 0')
            expected = torch.arange(seq.shape[1])[None, :] < valid.sum(1)[:, None]
            if not torch.equal(valid, expected):
                raise ValueError(f'{name}: padding must be contiguous on the right')
            raw = raw.flatten().float()
            if len(raw) != len(seq) or not torch.isfinite(raw).all() or not (raw > 0).all():
                raise ValueError(f'{name}: target must contain positive finite raw QoR labels')
            groups = {key: [] for key in ('train', 'validation', 'test')}
            for index, row in enumerate(seq):
                groups[recipe_partition(row[row.ne(0)].tolist(), seed)].append(index)
            if len(groups['train']) < 8 or len(groups['validation']) < 3:
                raise ValueError(f'{name}: need at least 8 train and 3 validation samples after hashing')
            groups = {key: torch.tensor(rows, dtype=torch.long) for key, rows in groups.items()}
            train_y = raw[groups['train']].double()
            mean, std = float(train_y.mean()), float(train_y.std(unbiased=False))
            if std < 1e-8:
                std = 1.0
            self.items.append(dict(id=name, x=x.long(), edge_index=edges.long(), seq=seq.long(),
                                   raw=raw, y=(raw - mean) / std, mean=mean, std=std, splits=groups))
            records.append(dict(id=name, file=spec['file'], sha256=file_hash(path), mean=mean, std=std,
                                split_indices={key: rows.tolist() for key, rows in groups.items()}))
        if not self.items:
            raise ValueError('Dataset contains no circuits')
        self.metadata = dict(format_version=manifest['format_version'], target=target, seed=seed,
                             split='global recipe SHA256 80/10/10; known circuits', circuits=records)
        self.fingerprint = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()


def graph_on_device(item, device):
    return Data(x=item['x'].to(device), edge_index=item['edge_index'].to(device),
                batch=torch.zeros(len(item['x']), dtype=torch.long, device=device))


def trim(seq):
    return seq[:, :int(seq.ne(0).sum(1).max())]
