import argparse
import csv
import json
import os
import sys
from pathlib import Path

import lightgbm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.predict import GUALOPredictor
from inference.metrics import mape, ranking, uncertainty
from integrity import digest, file_checksum

GROUPS = {
    'mape': ['i2c', 'spi', 'des3_area', 'ss_pcm', 'usb_phy', 'wb_dma', 'sasc', 'simple_spi', 'dynamic_node'],
    'ranking': ['wb_conmax', 'tinyRocket', 'aes_xcrypt', 'aes_secworks', 'jpeg', 'bp_be', 'ethernet', 'vga_lcd', 'picosoc', 'dft', 'idft', 'fir', 'iir', 'sha256'],
}
GROUPS['uq'] = sorted(GROUPS['mape'] + GROUPS['ranking'])


def main(default_task='all'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT / 'GUALO_test_demo.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets' / 'opencores')
    parser.add_argument('--task', choices=['all', 'mape', 'ranking', 'uq'], default=default_task)
    parser.add_argument('--designs', nargs='+')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if not args.model.is_file():
        parser.error('Place GUALO_test_demo.pt in the repository root, or specify --model.')
    torch.set_num_threads(args.threads)
    tasks = ['mape', 'ranking', 'uq'] if args.task == 'all' else [args.task]
    groups = {task: list(dict.fromkeys(args.designs or GROUPS[task])) for task in tasks}
    names = list(dict.fromkeys(name for task in tasks for name in groups[task]))
    if any(name not in GROUPS['uq'] for name in names):
        parser.error('Select a circuit from the supplied OpenCores dataset.')
    protocol = json.loads((ROOT / 'inference' / 'uq_indices.json').read_text())
    model_id = file_checksum(args.model)
    cache_dir = args.output / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    predictor = None
    rows = {task: [] for task in tasks}
    for name in names:
        file = args.data / f'OpenCores__{name}.pt'
        dataset_id = file_checksum(file)
        if dataset_id != protocol['designs'][name]['checksum']:
            raise ValueError(f'{name}: dataset differs from the supplied benchmark')
        settings = dict(format=1, model=model_id, data=dataset_id, torch=torch.__version__, numpy=np.__version__, lightgbm=lightgbm.__version__, device=args.device)
        key = digest(json.dumps(settings, sort_keys=True).encode())
        cached = cache_dir / f'{name}_{key}.npz'
        data = torch.load(file, map_location='cpu', weights_only=True)
        truth = {target: data[target].flatten().numpy().astype(np.float64) for target in ['area', 'delay']}
        n = len(data['opt_seq'])
        if n != protocol['designs'][name]['recipes']:
            raise ValueError(f'{name}: recipe count mismatch')
        if any(len(y) != n or not np.isfinite(y).all() or (y <= 0).any() for y in truth.values()):
            raise ValueError(f'{name}: labels must be positive and finite')
        if cached.exists() and not args.refresh:
            with np.load(cached, allow_pickle=False) as saved:
                output = {key: saved[key] for key in saved.files}
        else:
            if predictor is None:
                predictor = GUALOPredictor(args.model, args.device)
            output = predictor.predict(name, data)
            temporary = cached.with_name(cached.name + f'.{os.getpid()}.tmp.npz')
            np.savez_compressed(temporary, **output)
            temporary.replace(cached)
        for target in ['area', 'delay']:
            for suffix in ['', '_risk', '_risk_score']:
                values = output[target + suffix]
                if values.shape != (n,) or not np.isfinite(values).all():
                    raise ValueError(f'{name}: invalid {target + suffix} output')
        for task in tasks:
            if name not in groups[task]:
                continue
            record = dict(design=name, recipes=n)
            for target in ['area', 'delay']:
                if task == 'mape':
                    values = mape(output[target], truth[target])
                elif task == 'ranking':
                    values = ranking(output[target], truth[target])
                else:
                    values = uncertainty(output[target], truth[target], output[target + '_risk_score'], output[target + '_risk'], float(output[target + '_scale']), protocol['designs'][name]['indices'], protocol['candidate_fraction'], protocol['underestimation_std'])
                record.update({target + '_' + key: value for key, value in values.items()})
            rows[task].append(record)
        print(f'{name}: complete', file=sys.stderr, flush=True)
    summaries = {}
    for task, records in rows.items():
        fields = ['mape_percent'] if task == 'mape' else (['spearman', 'top10_percent'] if task == 'ranking' else ['auc'])
        summary = dict(circuits=len(records))
        for target in ['area', 'delay']:
            for field in fields:
                key = target + '_' + field
                values = [row[key] for row in records if row[key] is not None]
                summary[key] = float(np.mean(values)) if values else None
                summary[key + '_count'] = len(values)
        summaries[task] = summary
        with (args.output / f'{task}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        result = dict(summary=summary, per_design=records, mc_samples=90, seeds=[42, 123, 2026], aggregation='Unweighted circuit mean; undefined metrics excluded.')
        if task == 'uq':
            result['protocol'] = dict(candidate_fraction=protocol['candidate_fraction'], underestimation_std=protocol['underestimation_std'], score='raw directional risk', subset='uq_indices.json')
        (args.output / f'{task}.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    if args.task == 'all':
        (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    print(f'{"Metric":<24}{"Area":>12}{"Delay":>12}')
    for task, field, label, precision in [('mape', 'mape_percent', 'MAPE (%)', 2), ('ranking', 'spearman', 'Spearman', 3), ('ranking', 'top10_percent', 'Top-10% (%)', 2), ('uq', 'auc', 'Risk AUROC', 3)]:
        if task not in summaries:
            continue
        values = [summaries[task][t + '_' + field] for t in ['area', 'delay']]
        display = [('N/A' if x is None else f'{x:.{precision}f}') for x in values]
        print(f'{label:<24}{display[0]:>12}{display[1]:>12}')
    return summaries


if __name__ == '__main__':
    main()
