import json, math, os, random, sys, threading, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
OPS = {1: 'refactor', 2: 'refactor -z', 3: 'refactor -l', 4: 'refactor -l -z', 5: 'rewrite', 6: 'rewrite -z', 7: 'rewrite -l', 8: 'rewrite -l -z', 9: 'resub', 10: 'resub -z', 11: 'resub -l', 12: 'resub -l -z', 13: 'balance'}
RESYN2 = 'balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance'

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    tmp.replace(path)

def costs(predictions, reference, kappa):
    return [[(p[t + '_mu'] + kappa * p[t + '_u']) / reference[j] for j, t in enumerate(('area', 'delay'))] for p in predictions]

def front_rows(rows):
    result, lowest = ([], math.inf)
    for row in sorted(rows, key=lambda r: (r['delay'], r['area'])):
        if row['area'] < lowest:
            result.append(row)
            lowest = row['area']
    return result

class BudgetEnded(Exception):
    pass

class SharedPredictor:

    def __init__(self, engine, out):
        self.engine, self.out = (engine, out)
        self.lock = threading.Lock()
        self.cache = {}

    def predict(self, design, seqs, remaining):
        with self.lock:
            cache = self.cache.setdefault(design, {})
            missing = list(dict.fromkeys((tuple(s) for s in seqs if tuple(s) not in cache)))
            for offset in range(0, len(missing), 128):
                if remaining() <= 0:
                    raise BudgetEnded()
                batch = missing[offset:offset + 128]
                data = self.engine.predict('OpenCores__' + design, [list(s) for s in batch])
                for seq, pred in zip(batch, data['rows']):
                    cache[seq] = pred
                with (self.out / 'predictions.jsonl').open('a') as f:
                    for seq in batch:
                        f.write(json.dumps(dict(design=design, **cache[seq])) + '\n')
            return [cache[tuple(s)] for s in seqs]

def dataset_path(directory, design):
    path = directory / f'OpenCores__{design}.pt'
    return path if path.is_file() else directory / f'{design}.pt'

class Predictor:

    def __init__(self, model, dataset_dir, designs, inference_root, device=None):
        sys.path.insert(0, str(inference_root))
        from inference.predict import GUALOPredictor
        self.lock = threading.Lock()
        self.model = GUALOPredictor(model, device=device)
        self.data = {}
        self.pools = {}
        for d in designs:
            blob = torch.load(dataset_path(dataset_dir, d), weights_only=True, map_location='cpu')
            for target, engine in self.model.engines.items():
                if d not in engine.risk.settings['designs']:
                    raise ValueError(f'Model has no {target} risk module for {d}')
            self.data[d] = {'x': blob['x'], 'edge_index': blob['edge_index']}
            self.pools[d] = list(dict.fromkeys((tuple(row[row.ne(0)].tolist()) for row in blob['opt_seq'])))

    def predict(self, design, seqs):
        d = design.removeprefix('OpenCores__')
        with self.lock:
            tokens = torch.zeros((len(seqs), max(map(len, seqs))), dtype=torch.long)
            for i, s in enumerate(seqs):
                tokens[i, :len(s)] = torch.tensor(s)
            result = self.model.predict(d, dict(self.data[d], opt_seq=tokens))
            rows = []
            for i, s in enumerate(seqs):
                row = {'seq': s}
                for t in ['area', 'delay']:
                    mu = float(result[t][i])
                    score = float(result[t + '_cost_score'][i])
                    row[t + '_mu'] = mu
                    row[t + '_u'] = max(0.0, score - mu)
                    row[t + '_risk'] = float(result[t + '_risk'][i])
                rows.append(row)
            return {'rows': rows}

class PoolMixin:
    pools = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_pool = list(self.pools[self.design])
        random.Random(self.config['seed']).shuffle(self.source_pool)
        self.cursor = 0

    def fresh(self, n):
        out = []
        while len(out) < n and self.cursor < len(self.source_pool):
            if self.remaining() <= 0:
                raise BudgetEnded()
            seq = self.source_pool[self.cursor]
            self.cursor += 1
            if seq not in self.seen:
                self.seen.add(seq)
                out.append(seq)
        if len(out) < n:
            out.extend(super().fresh(n - len(out)))
        return out

def measure(rows, reference):
    if not rows:
        return dict(abc_success=0, best_q_x100=None, front=[])
    return dict(abc_success=len(rows), best_q_x100=100 * max((2 - r['area'] / reference[0] - r['delay'] / reference[1] for r in rows)), front=front_rows(rows))

def snapshot(search, seconds=None):
    rows = []
    path = search.out / 'abc.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if seconds is None or r['elapsed'] <= seconds:
                rows.append(r)
    result = dict(design=search.design, method=search.method, seed=search.config['seed'], status=search.status, error=search.error, phase=search.phase, generation=search.generation, abc_calls=search.abc_calls, abc_failures=search.failed_calls, predicted=search.predicted, reference=search.reference)
    if search.reference:
        result.update(measure(rows, search.reference))
    return result
