import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import torch
from common import ROOT, Predictor, PoolMixin, SharedPredictor, RESYN2, save, snapshot, BudgetEnded, dataset_path
from abc_runner import ABCOracle
from nsga2 import Baseline
from search import GUALOSearch
from plot import plot_results

def checksum(path):
    h = hashlib.blake2b(digest_size=32)
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4194304), b''):
            h.update(part)
    return h.hexdigest()

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, help='Default: datasets/opencores under the inference repository.')
    p.add_argument('--abc', type=Path, required=True)
    p.add_argument('--liberty', type=Path, required=True)
    p.add_argument('--aig-dir', type=Path, required=True)
    p.add_argument('--inference-root', type=Path, default=ROOT.parent)
    p.add_argument('--designs', nargs='+', default=['bp_be'])
    p.add_argument('--methods', nargs='+', choices=['nsga2', 'lcb'], default=['nsga2', 'lcb'])
    p.add_argument('--device', default=None)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261122)
    p.add_argument('--seconds', type=float, default=3600)
    p.add_argument('--max-generations', type=int, default=0)
    p.add_argument('--initial-pool', type=int, default=1024)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--injection-pool', type=int, default=128)
    p.add_argument('--abc-workers', type=int, default=4)
    p.add_argument('--preflight-checks', type=Path)
    args = p.parse_args()
    if args.data is None:
        args.data = args.inference_root / 'datasets' / 'opencores'
    if args.seconds <= 0 or args.initial_pool < 32 or args.abc_workers < 1 or (args.max_generations < 0):
        p.error('Use positive budgets, nonnegative generations, and candidate pools of at least 32.')
    if not 0 <= args.alpha <= 1 or args.injection_pool < round(32 * args.alpha):
        p.error('Invalid injection settings.')
    if len(set(args.methods)) != len(args.methods) or len(set(args.designs)) != len(args.designs):
        p.error('Methods and designs must be unique.')
    if not all((re.fullmatch('[A-Za-z0-9_]+', d) for d in args.designs)):
        p.error('Design names may contain letters, digits, and underscores.')
    for name in ['model', 'data', 'abc', 'liberty', 'aig_dir', 'inference_root', 'out', 'preflight_checks']:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    files = [args.model, args.abc, args.liberty]
    files += [args.aig_dir / (d + '.aig') for d in args.designs]
    files += [dataset_path(args.data, d) for d in args.designs]
    if any((m != 'nsga2' for m in args.methods)):
        files.append(args.inference_root / 'inference/predict.py')
    files += [f for f in [args.preflight_checks] if f is not None]
    for path in files:
        if not path.is_file():
            p.error(f'File not found: {path}')
    if not os.access(args.abc, os.X_OK):
        p.error('ABC must be executable.')
    if args.out.exists():
        p.error('Output directory already exists; choose a new directory.')
    return args

def main():
    args = arguments()
    torch.set_num_threads(2)
    if args.preflight_checks:
        checked = json.loads(args.preflight_checks.read_text())
        for design in args.designs:
            rows = [r for r in checked if r['design'] == design]
            if not rows or any((r['max_error'] >= 0.03 for r in rows)):
                raise ValueError(f'ABC parity check failed or missing: {design}')
    args.out.mkdir(parents=True)
    model_id = checksum(args.model)
    save(args.out / 'status.json', dict(phase='loading_models', pid=os.getpid()))
    start = time.monotonic()
    engine = None
    if any((m != 'nsga2' for m in args.methods)):
        engine = Predictor(args.model, args.data, args.designs, args.inference_root, args.device)
        PoolMixin.pools = engine.pools
    else:
        PoolMixin.pools = {}
        for design in args.designs:
            blob = torch.load(dataset_path(args.data, design), weights_only=True, map_location='cpu')
            PoolMixin.pools[design] = list(dict.fromkeys((tuple(row[row.ne(0)].tolist()) for row in blob['opt_seq'])))
    load_seconds = time.monotonic() - start
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(model_id=model_id, population=32, model_load_seconds=load_seconds, timing='Shared start and deadline; inference, queue waiting, selection, ABC and IO included; model loading excluded.', input_ids={'abc': checksum(args.abc), 'liberty': checksum(args.liberty), 'aigs': {d: checksum(args.aig_dir / (d + '.aig')) for d in args.designs}})
    save(args.out / 'protocol.json', protocol)
    abc = ABCOracle(args.abc, args.liberty, args.aig_dir)
    searches = []
    with ThreadPoolExecutor(max_workers=args.abc_workers) as oracle:

        def limited_abc(name, seq, script, timeout):
            deadline = time.monotonic() + timeout

            def execute():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BudgetEnded()
                return abc(name, seq, script, remaining)
            return oracle.submit(execute).result()
        for design in args.designs:
            for method in args.methods:
                out = args.out / design / method
                out.mkdir(parents=True)
                config = dict(population=32, seed=args.seed, evolution_seed=args.seed + 100000, seconds=args.seconds, abc_workers=1, initial_pool=32 if method == 'nsga2' else args.initial_pool, injection_pool=args.injection_pool, alpha=args.alpha, max_generations=args.max_generations, kappa=1, arm=method, resyn2=RESYN2)
                cls = Baseline if method == 'nsga2' else GUALOSearch
                pred = None if method == 'nsga2' else SharedPredictor(engine, out)
                searches.append(cls(config, design, method, pred, set(), out, args.out / 'live', limited_abc))
        gate = threading.Event()
        clock = {}
        milestones = {}

        def work(search):
            gate.wait()
            search.start = clock['mono']
            search.started_at = clock['wall']
            search.run()
        with ThreadPoolExecutor(max_workers=len(searches)) as pool:
            futures = [pool.submit(work, s) for s in searches]
            clock.update(mono=time.monotonic(), wall=time.time())
            save(args.out / 'timing.json', dict(started_at=clock['wall'], deadline=clock['wall'] + args.seconds))
            gate.set()
            while True:
                elapsed = time.monotonic() - clock['mono']
                rows = []
                for s in searches:
                    row = snapshot(s)
                    rows.append(row)
                save(args.out / 'status.json', dict(phase='running', pid=os.getpid(), elapsed=elapsed, seconds=args.seconds, arms=rows))
                for point in sorted({3600, 5400, 7200, 18000, args.seconds}):
                    if elapsed >= point and str(point) not in milestones:
                        milestones[str(point)] = [snapshot(s, point) for s in searches]
                        save(args.out / 'milestones.json', milestones)
                if all((f.done() for f in futures)):
                    for f in futures:
                        f.result()
                    break
                time.sleep(1)
    results = [snapshot(s, args.seconds) for s in searches]
    save(args.out / 'results.json', results)
    for point in sorted({3600, 5400, 7200, 18000, args.seconds}):
        if args.seconds >= point:
            milestones[str(point)] = [snapshot(s, point) for s in searches]
    save(args.out / 'milestones.json', milestones)
    if checksum(args.model) != model_id:
        raise RuntimeError('Model file changed during execution.')
    failed = any((s.status != 'complete' for s in searches))
    save(args.out / 'status.json', dict(phase='failed' if failed else 'complete', pid=os.getpid(), completed_at=time.time(), arms=results, model_unchanged=True))
    if failed:
        raise RuntimeError('An experiment arm failed; see status.json.')
    print(args.out / 'results.json', flush=True)
    for path in plot_results(args.out):
        print(path, flush=True)
if __name__ == '__main__':
    main()
