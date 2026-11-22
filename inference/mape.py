import argparse
import csv
import gc
import json
import math
from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import GUALO, MODEL_VERSION, enable_dropout
from integrity import stable_seed


GROUPS = {
    'mape': ['i2c', 'spi', 'des3_area', 'ss_pcm', 'usb_phy', 'wb_dma',
             'sasc', 'simple_spi', 'dynamic_node'],
    'ranking': ['wb_conmax', 'tinyRocket', 'aes_xcrypt', 'aes_secworks',
                'jpeg', 'bp_be', 'ethernet', 'vga_lcd', 'picosoc', 'dft',
                'idft', 'fir', 'iir', 'sha256'],
}


def arguments(default_group):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets' / 'opencores')
    parser.add_argument('--area-checkpoint', type=Path)
    parser.add_argument('--delay-checkpoint', type=Path)
    parser.add_argument('--group', choices=['mape', 'ranking', 'all'], default=default_group)
    parser.add_argument('--designs', nargs='+', help='Circuit names; overrides --group.')
    parser.add_argument('--mc-samples', type=int, default=30, help='Dropout passes; 0 disables dropout.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--mc-chunk', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, help='Optional per-circuit CSV output.')
    args = parser.parse_args()
    if args.mc_samples < 0 or min(args.batch_size, args.mc_chunk, args.threads) < 1:
        parser.error('Sample count must be nonnegative; batch size, chunk size and threads must be positive.')
    return args


def selected_files(args):
    names = args.designs or (sum(GROUPS.values(), []) if args.group == 'all' else GROUPS[args.group])
    paths = []
    for name in dict.fromkeys(names):
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError(f'Invalid circuit name: {name}')
        stem = name if name.startswith('OpenCores__') else 'OpenCores__' + name
        path = args.data / (stem + '.pt')
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def evaluate(args, metric):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    files = selected_files(args)
    rows = {path.stem: dict(design=path.stem.removeprefix('OpenCores__')) for path in files}
    checkpoints = {}
    for target in ['area', 'delay']:
        path = getattr(args, target + '_checkpoint')
        if path is None:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id='Alex959/GUALO', filename=target + '_demo.pt'))
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('model_version') != MODEL_VERSION:
            raise ValueError(f'{target}: incompatible model version')
        if checkpoint.get('settings', {}).get('target') != target:
            raise ValueError(f'{target}: checkpoint target does not match')
        norms = {r['id']: r for r in checkpoint['normalizations']}
        model = GUALO(**checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        checkpoints[target] = dict(path=str(path), epoch=checkpoint.get('epoch'))
        del checkpoint
        with torch.inference_mode():
            for data_path in files:
                did = data_path.stem
                if did not in norms:
                    raise ValueError(f'{target}: checkpoint has no normalization for {did}')
                normal = norms[did]
                if not math.isfinite(normal['mean']) or not math.isfinite(normal['std']) or normal['std'] <= 0:
                    raise ValueError(f'{target}: invalid normalization for {did}')
                blob = torch.load(data_path, map_location='cpu', weights_only=True)
                tokens = blob['opt_seq'].long()
                truth = blob[target].flatten().double()
                if tokens.ndim != 2 or len(tokens) != len(truth) or len(truth) < 2:
                    raise ValueError(f'{did}: invalid recipe or label dimensions')
                valid = tokens.ne(0)
                padding = torch.arange(tokens.shape[1])[None, :] < valid.sum(1)[:, None]
                vocabulary = model.synth_transformer.embedding.num_embeddings
                if not valid.any(1).all() or not torch.equal(valid, padding) or not ((tokens >= 0) & (tokens < vocabulary)).all():
                    raise ValueError(f'{did}: invalid recipe tokens or padding')
                if not torch.isfinite(truth).all() or not (truth > 0).all():
                    raise ValueError(f'{did}: labels must be finite and positive')
                if 'recipes' in rows[did] and rows[did]['recipes'] != len(truth):
                    raise ValueError(f'{did}: inconsistent recipe counts')
                rows[did]['recipes'] = len(truth)
                graph = Data(x=blob['x'].long().to(device), edge_index=blob['edge_index'].long().to(device),
                             batch=torch.zeros(len(blob['x']), dtype=torch.long, device=device))
                model.eval()
                embedding = model.gnn(graph)
                if args.mc_samples:
                    enable_dropout(model)
                seed = stable_seed(f'{args.seed}:{target}:{did}'.encode(), 2**31)
                devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
                outputs = []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    for batch in tokens.split(args.batch_size):
                        batch = batch[:, :int(batch.ne(0).sum(1).max())].to(device)
                        samples = []
                        passes = max(1, args.mc_samples)
                        for offset in range(0, passes, args.mc_chunk):
                            count = min(args.mc_chunk, passes - offset)
                            repeated = batch.repeat((count, 1))
                            output = model.transformer.forward_encoded_graph(embedding.expand(len(repeated), -1), repeated, False)
                            mu, _ = model._split_prediction(output)
                            samples.append(mu.reshape(count, len(batch)).cpu())
                        outputs.append(torch.cat(samples).mean(0))
                pred = torch.cat(outputs).double() * normal['std'] + normal['mean']
                if not torch.isfinite(pred).all():
                    raise ValueError(f'{did}: predictions contain non-finite values')
                rows[did].update({target + '_' + key: value for key, value in metric(pred, truth).items()})
                del graph, embedding, blob, tokens, truth, pred, outputs
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    records = list(rows.values())
    metrics = [key for key in records[0] if key not in {'design', 'recipes'}]
    summary = {}
    for key in metrics:
        values = [row[key] for row in records if row[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
    result = dict(checkpoints=checkpoints, data=str(args.data), mc_samples=args.mc_samples,
                  seed=args.seed, circuits=len(records), recipes=sum(r['recipes'] for r in records),
                  aggregation='unweighted mean of per-circuit metrics; undefined values excluded',
                  summary=summary, per_design=records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def mape(predicted, actual):
    return {'mape_percent': float(((predicted - actual).abs() / actual).mean() * 100)}


if __name__ == '__main__':
    evaluate(arguments('mape'), mape)
