import hashlib, json, math, random, time, traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from common import BudgetEnded, RESYN2, PoolMixin, front_rows, save

def ranks_and_crowding(values):
    v = np.asarray(values, dtype=float)
    assert v.ndim == 2 and v.shape[1] == 2 and np.isfinite(v).all() and len(v)
    dominates = (v[:, None, :] <= v[None, :, :]).all(2) & (v[:, None, :] < v[None, :, :]).any(2)
    incoming = dominates.sum(0)
    ranks = np.full(len(v), -1, dtype=int)
    crowd = np.zeros(len(v))
    front = np.flatnonzero(incoming == 0).tolist()
    rank = 0
    while front:
        ranks[front] = rank
        for axis in range(2):
            order = sorted(front, key=lambda i: (v[i, axis], i))
            spread = v[order[-1], axis] - v[order[0], axis]
            if spread <= 1e-12:
                continue
            crowd[order[0]] = crowd[order[-1]] = np.inf
            for j in range(1, len(order) - 1):
                crowd[order[j]] += (v[order[j + 1], axis] - v[order[j - 1], axis]) / spread
        following = []
        for i in front:
            for j in np.flatnonzero(dominates[i]):
                incoming[j] -= 1
                if incoming[j] == 0:
                    following.append(int(j))
        front, rank = (following, rank + 1)
    assert (ranks >= 0).all()
    return (ranks, crowd)

def selection(values, n):
    ranks, crowd = ranks_and_crowding(values)
    return sorted(range(len(values)), key=lambda i: (ranks[i], -crowd[i], i))[:n]

def fast_ranks_and_crowding(values):
    v = np.asarray(values, dtype=float)
    assert v.ndim == 2 and v.shape[1] == 2 and len(v) and np.isfinite(v).all()
    compressed = np.searchsorted(np.unique(v[:, 1]), v[:, 1]) + 1
    tree = np.full(int(compressed.max()) + 1, -1, dtype=int)
    order = sorted(range(len(v)), key=lambda i: (v[i, 0], v[i, 1], i))
    ranks = np.zeros(len(v), dtype=int)
    crowd = np.zeros(len(v))
    offset = 0
    while offset < len(order):
        end = offset + 1
        i = order[offset]
        while end < len(order) and np.array_equal(v[order[end]], v[i]):
            end += 1
        pos = int(compressed[i])
        best = -1
        while pos > 0:
            best = max(best, int(tree[pos]))
            pos -= pos & -pos
        rank = best + 1
        for j in order[offset:end]:
            ranks[j] = rank
        pos = int(compressed[i])
        while pos < len(tree):
            tree[pos] = max(tree[pos], rank)
            pos += pos & -pos
        offset = end
    fronts = {}
    for i, r in enumerate(ranks):
        fronts.setdefault(int(r), []).append(i)
    for front in fronts.values():
        for axis in range(2):
            seq = sorted(front, key=lambda i: (v[i, axis], i))
            spread = v[seq[-1], axis] - v[seq[0], axis]
            if spread <= 1e-12:
                continue
            crowd[seq[0]] = crowd[seq[-1]] = np.inf
            for j in range(1, len(seq) - 1):
                crowd[seq[j]] += (v[seq[j + 1], axis] - v[seq[j - 1], axis]) / spread
    return (ranks, crowd)

def fast_selection(values, n):
    rank, crowd = fast_ranks_and_crowding(values)
    return sorted(range(len(rank)), key=lambda i: (rank[i], -crowd[i], i))[:n]

class SearchBase:

    def __init__(self, config, design, method, predictor, excluded, out, public, abc):
        self.config, self.design, self.method = (config, design, method)
        self.predictor, self.excluded, self.abc = (predictor, excluded, abc)
        self.id = f'{design}_{method}'
        self.out = out / self.id
        self.out.mkdir(parents=True, exist_ok=True)
        self.public = public / (self.id + '.json')
        self.rng = random.Random(config['seed'])
        self.seen, self.truth, self.scored = (set(), {}, set())
        self.pop, self.history, self.events = ([], [], [])
        self.generation, self.abc_calls, self.failed_calls, self.predicted = (0, 0, 0, 0)
        self.reference = None
        self.candidates, self.counts, self.phase = ([], {}, 'starting')
        self.selected = []
        self.start = time.monotonic()
        self.started_at = time.time()
        self.status, self.error = ('running', None)
        self.kappa = config['kappa'] if method == 'lcb' else 0.0
        self.workers = ThreadPoolExecutor(max_workers=config['abc_workers'])
        self._front = []
        self._best_q = None

    def remaining(self):
        return self.config['seconds'] - (time.monotonic() - self.start)

    def fresh(self, n):
        seqs = []
        while len(seqs) < n:
            if self.remaining() <= 0:
                raise BudgetEnded()
            length = max(1, min(20, round(self.rng.gauss(10, 4))))
            seq = tuple((self.rng.randint(1, 13) for _ in range(length)))
            if seq not in self.excluded and seq not in self.seen:
                self.seen.add(seq)
                seqs.append(seq)
        return seqs

    def publish(self, phase=None, **counts):
        self.counts.update(counts)
        if phase:
            self.phase = phase
            self.events.append(dict(phase=phase, generation=self.generation, elapsed=time.monotonic() - self.start, counts=dict(self.counts)))
            self.events = self.events[-40:]
        data = dict(id=self.id, design=self.design, method=self.method, status=self.status, phase=self.phase, generation=self.generation, started_at=self.started_at, updated_at=time.time(), elapsed=time.monotonic() - self.start, budget_seconds=self.config['seconds'], abc_calls=self.abc_calls, abc_succeeded=len(self.truth), abc_failed=self.failed_calls, baseline_calls=int(self.reference is not None), predicted=self.predicted, best_q=self._best_q, front=self._front, history=self.history, counts=self.counts, events=self.events, candidates=self.candidates, population=[list(s) for s in self.pop], reference=self.reference, config=self.config, error=self.error)
        save(self.public, data)

    def predict(self, seqs):
        values = self.predictor.predict(self.design, seqs, self.remaining)
        self.scored.update(seqs)
        self.predicted = len(self.scored)
        return values

    def label(self, seqs):
        missing = [s for s in dict.fromkeys(seqs) if s not in self.truth]
        pending, cursor, done = ({}, 0, 0)
        self.publish('abc', abc_pending=len(missing), abc_done=0, abc_needed=len(missing))
        while cursor < len(missing) or pending:
            while cursor < len(missing) and len(pending) < self.config['abc_workers'] and (self.remaining() > 0):
                seq = missing[cursor]
                cursor += 1
                self.abc_calls += 1
                f = self.workers.submit(self.abc, self.design, seq, None, min(180.0, self.remaining()))
                pending[f] = seq
            if not pending:
                raise BudgetEnded()
            ready, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            for f in ready:
                seq = pending.pop(f)
                try:
                    area, delay = f.result()
                except Exception:
                    self.failed_calls += 1
                    if self.remaining() <= 0:
                        continue
                    raise
                row = dict(id=self.design + '-' + hashlib.blake2b(bytes(seq), digest_size=6).hexdigest(), seq=list(seq), area=area, delay=delay)
                self.truth[seq] = row
                with (self.out / 'abc.jsonl').open('a') as log:
                    log.write(json.dumps(dict(**row, elapsed=time.monotonic() - self.start)) + '\n')
                done += 1
            self.publish(abc_pending=len(missing) - done, abc_done=done)
        if any((s not in self.truth for s in missing)):
            raise BudgetEnded()

    def evolve(self):
        n = self.config['population']
        rank, crowd = ranks_and_crowding([[self.truth[s]['area'], self.truth[s]['delay']] for s in self.pop])

        def parent():
            a, b = (self.rng.randrange(n), self.rng.randrange(n))
            return self.pop[min((a, b), key=lambda i: (rank[i], -crowd[i], i))]
        children = []
        while len(children) < n:
            if self.remaining() <= 0:
                raise BudgetEnded()
            a, b = (parent(), parent())
            cut_a, cut_b = (self.rng.randrange(len(a) + 1), self.rng.randrange(len(b) + 1))
            seq = list((a[:cut_a] + b[cut_b:])[:20]) or [self.rng.randint(1, 13)]
            seq = [self.rng.randint(1, 13) if self.rng.random() < 1 / len(seq) else t for t in seq]
            if len(seq) < 20 and self.rng.random() < 0.1:
                seq.insert(self.rng.randrange(len(seq) + 1), self.rng.randint(1, 13))
            if len(seq) > 1 and self.rng.random() < 0.1:
                del seq[self.rng.randrange(len(seq))]
            seq = tuple(seq)
            if seq not in self.excluded and seq not in self.seen:
                self.seen.add(seq)
                children.append(seq)
        return children

    def record(self, complete=True):
        rows = list(self.truth.values())
        self._front = front_rows(rows)
        self._best_q = max((2 - r['area'] / self.reference[0] - r['delay'] / self.reference[1] for r in rows), default=None)
        self.history.append(dict(generation=self.generation, elapsed=time.monotonic() - self.start, abc_calls=self.abc_calls, abc_succeeded=len(rows), best_q=self._best_q, complete_generation=complete))
        save(self.out / f'generation_{self.generation:06d}.json', dict(generation=self.generation, population=[list(s) for s in self.pop], candidates=self.candidates, counts=self.counts, last=self.history[-1], rng_state=self.rng.getstate(), events=[e for e in self.events if e['generation'] == self.generation]))
        self.publish('update')

class PilotMixin:

    def publish(self, phase=None, **counts):
        original = self.candidates
        self.candidates = original[:96]
        try:
            super().publish(phase, **counts)
        finally:
            self.candidates = original

    def record(self, complete=True):
        if self.generation == 0 and (not getattr(self, 'evolution_seed_set', False)):
            self.rng = random.Random(self.config['evolution_seed'])
            self.evolution_seed_set = True
        super().record(complete)
        if self.generation == 0 and complete:
            rows = [self.truth[s] for s in self.pop]
            initial = dict(design=self.design, arm=self.config['arm'], initial_pool=self.config['initial_pool'], population=len(rows), measured_q_mean=float(np.mean([2 - r['area'] / self.reference[0] - r['delay'] / self.reference[1] for r in rows])), **self.history[-1])
            if self.predictor:
                preds = [self.predictor.cache[self.design][s] for s in self.pop]
                initial['selected_recipe_mape'] = {t: float(np.mean([abs(p[t + '_mu'] - r[t]) / r[t] * 100 for p, r in zip(preds, rows)])) for t in ['area', 'delay']}
            save(self.out / 'initial_result.json', initial)
            save(self.public.parent / (self.id + '_initial.json'), initial)

class NSGA2Search(SearchBase):

    def run(self):
        n = self.config['population']
        try:
            if self.remaining() <= 0:
                raise BudgetEnded()
            self.publish('baseline')
            self.reference = self.abc(self.design, None, RESYN2, min(180.0, self.remaining()))
            self.publish('initial', initial_pool=n, survivors=n)
            initial = self.fresh(n)
            self.predicted = len(self.seen)
            self.label(initial)
            self.pop = initial
            self.record()
            while self.remaining() > 0 and (not self.config.get('max_generations') or self.generation < self.config['max_generations']):
                self.generation += 1
                self.counts = dict(parents=n, offspring=n, injected=0, survivors=n)
                self.publish('variation')
                children = self.evolve()
                self.predicted = len(self.seen)
                self.label(children)
                merged = self.pop + children
                self.publish('merge', merged=len(merged))
                keep = selection([[self.truth[s]['area'], self.truth[s]['delay']] for s in merged], n)
                self.selected = [merged[i] for i in keep]
                self.candidates = [dict(seq=list(s), origin='parent' if i < n else 'offspring', area=self.truth[s]['area'], delay=self.truth[s]['delay'], selected=i in keep) for i, s in enumerate(merged)]
                self.publish('select')
                self.pop = list(self.selected)
                self.record()
            self.status = 'complete'
        except BudgetEnded:
            self.status = 'complete'
        except Exception as exc:
            self.status, self.error = ('failed', f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
        finally:
            self.workers.shutdown(wait=True)
            if self.reference is not None and self.truth and (not self.history or len(self.truth) > self.history[-1]['abc_succeeded']):
                self.record(complete=False)
            self.publish('finished' if self.status == 'complete' else 'failed')
            print(json.dumps(dict(id=self.id, status=self.status, generation=self.generation, abc_calls=self.abc_calls, error=self.error)), flush=True)

class Baseline(PoolMixin, PilotMixin, NSGA2Search):
    pass
