import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import torch
import torch.nn.functional as F

from checkpointing import atomic_json, sha
from data import CircuitDataset, graph_on_device, trim
from finetune.objectives import pairwise_loss, uncertainty_loss
from losses import nll_loss
from model import GUALO, MODEL_VERSION, enable_dropout
from train import atomic_save, average, correlation, predict


def load_data(args, checkpoint):
    seed = checkpoint.get('settings', {}).get('seed', 42)
    dataset = CircuitDataset(args.data, args.target, seed)
    if args.design:
        dataset.items = [item for item in dataset.items if item['id'] == args.design]
        if not dataset.items:
            raise ValueError(f'Unknown design: {args.design}')
    norms = {row['id']: row for row in checkpoint.get('normalizations', [])}
    for item in dataset.items:
        if item['id'] in norms:
            reference = norms[item['id']]
            item['mean'], item['std'] = float(reference['mean']), float(reference['std'])
            if not math.isfinite(item['mean']) or not math.isfinite(item['std']) or item['std'] <= 0:
                raise ValueError('Invalid checkpoint normalization')
            item['y'] = (item['raw'] - item['mean']) / item['std']
        groups = {key: [] for key in ('select', 'calibrate', 'uq_validation')}
        for index in item['splits']['validation'].tolist():
            tokens = item['seq'][index]
            value = int.from_bytes(hashlib.sha256(bytes(tokens[tokens.ne(0)].tolist())
                                   + f':{seed}'.encode()).digest()[:8], 'big') % 1000
            key = 'select' if value < 850 else 'calibrate' if value < 875 else 'uq_validation'
            groups[key].append(index)
        item['splits'].update({key: torch.tensor(rows, dtype=torch.long) for key, rows in groups.items()})
        required = ('calibrate', 'uq_validation') if args.task == 'uq' else ('train', 'select')
        if any(len(item['splits'][key]) < 3 for key in required):
            raise ValueError(f'{item["id"]}: need at least three recipes in each of {required}')
    records = [dict(id=item['id'], mean=item['mean'], std=item['std'],
                    splits={key: rows.tolist() for key, rows in item['splits'].items()})
               for item in dataset.items]
    metadata = dict(source_fingerprint=dataset.fingerprint, partition_seed=seed, circuits=records,
                    buckets=dict(train=[0, 800], select=[800, 850], calibrate=[850, 875],
                                 uq_validation=[875, 900], test_excluded=[900, 1000]))
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return dataset.items, metadata, fingerprint


def configure(model, task, unfreeze_gnn):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads = [model.transformer.sm_mlp_head, model.transformer.lg_mlp_head]
    if task == 'uq':
        modules = [head[-1] for head in heads]
    elif task == 'ranking':
        modules = [model.transformer.multi_scale_encoder.layers[-1],
                   model.synth_transformer.transformer_encoder.layers[-1], *heads]
    else:
        modules = [model.transformer]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if not unfreeze_gnn:
        for parameter in model.gnn.parameters():
            parameter.requires_grad_(False)
    hooks = []
    if task in ('ranking', 'uq'):
        row = 0 if task == 'ranking' else 1
        def mask(gradient):
            result = torch.zeros_like(gradient)
            result[row] = gradient[row]
            return result
        for head in heads:
            for parameter in head[-1].parameters():
                hooks.append(parameter.register_hook(mask))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return parameters, hooks


def monte_carlo(model, embedding, seq, samples, seed):
    devices = [seq.device.index or 0] if seq.device.type == 'cuda' else []
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        enable_dropout(model)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            results = [predict(model, embedding, seq) for _ in range(samples)]
        means = torch.stack([value[0].flatten() for value in results])
        variances = torch.stack([value[1].flatten() for value in results])
        return means, variances
    finally:
        for module, training in modes:
            module.training = training


def embedding_for(model, item, device):
    if 'embedding' in item:
        return item['embedding'].to(device)
    return model.gnn(graph_on_device(item, device))


@torch.no_grad()
def evaluate(model, items, split, args, device):
    model.eval()
    records = []
    for slot, item in enumerate(items):
        embedding = embedding_for(model, item, device)
        indices = item['splits'][split]
        predictions, variances = [], []
        for batch, rows in enumerate(indices.split(args.batch_size)):
            seq = trim(item['seq'][rows]).to(device)
            if args.task == 'uq':
                means, log_variances = monte_carlo(model, embedding, seq, args.mc_samples,
                                                   args.seed + 100000 + slot * 10000 + batch)
                mu = means.mean(0)
                variance = means.var(0, unbiased=False) + log_variances.exp().mean(0)
            else:
                mu, log_var = predict(model, embedding, seq)
                mu, variance = mu.flatten(), log_var.flatten().exp()
            predictions.append(mu.cpu())
            variances.append(variance.cpu())
        mu, variance = torch.cat(predictions), torch.cat(variances).clamp_min(1e-8)
        y = item['y'][indices]
        error = (mu - y).abs()
        k = max(1, math.ceil(len(y) * 0.1))
        top = set(mu.argsort(stable=True)[:k].tolist())
        truth = set(y.argsort(stable=True)[:k].tolist())
        record = dict(id=item['id'], n=len(y),
                      mape=float((error * item['std'] / item['raw'][indices]).mean() * 100),
                      rank=correlation(mu, y, rank=True), top10_recall=len(top & truth) / k,
                      nll=float(0.5 * (variance.log() + error.square() / variance).mean()))
        if args.task == 'uq':
            count = max(1, math.ceil(len(y) * 0.2))
            high = variance.argsort(stable=True)[-count:]
            worst = error.argsort(stable=True)[-count:]
            record.update(uq=correlation(variance.sqrt(), error, rank=True),
                          risk20_recall=len(set(high.tolist()) & set(worst.tolist())) / count,
                          coverage90=float((error <= 1.6448536269514722 * variance.sqrt()).float().mean()),
                          retained80_error=float(error[variance.argsort(stable=True)[:-count]].mean()),
                          mean_error=float(error.mean()))
        records.append(record)
    keys = ('mape', 'rank', 'top10_recall', 'nll', 'uq', 'risk20_recall',
            'coverage90', 'retained80_error', 'mean_error')
    macro = {key: average([record.get(key) for record in records]) for key in keys}
    if any(not math.isfinite(value) for record in records for key, value in record.items()
           if key != 'id' and value is not None):
        raise RuntimeError('Nonfinite validation metrics')
    return dict(split=split, macro=macro, per_design=records)


def selection_score(metrics, baseline, args):
    current, reference = metrics['macro'], baseline['macro']
    if args.task == 'ranking':
        if current['mape'] > reference['mape'] * (1 + args.mape_tolerance):
            return None
        return (current['rank'] if current['rank'] is not None else -1.0, current['top10_recall'])
    if args.task == 'uq':
        if current['nll'] > reference['nll'] + args.nll_tolerance:
            return None
        return (current['uq'] if current['uq'] is not None else -1.0, -current['nll'])
    return (-current['mape'], current['rank'] if current['rank'] is not None else -1.0)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['ranking', 'uq', 'domain'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--target', choices=['area', 'delay'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--design')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float)
    p.add_argument('--rank-weight', type=float)
    p.add_argument('--mape-tolerance', type=float, default=0.05)
    p.add_argument('--nll-tolerance', type=float, default=0.05)
    p.add_argument('--mc-samples', type=int, default=30)
    p.add_argument('--unfreeze-gnn', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.lr = args.lr if args.lr is not None else (1e-3 if args.task == 'uq' else 1e-5)
    args.rank_weight = args.rank_weight if args.rank_weight is not None else (0.1 if args.task == 'uq' else 0.2)
    if min(args.epochs, args.threads) < 1 or args.batch_size < 2 or args.mc_samples < 2:
        raise ValueError('Require epochs/threads>=1, batch-size>=2 and mc-samples>=2')
    values = [args.lr, args.rank_weight, args.mape_tolerance, args.nll_tolerance]
    if not all(math.isfinite(value) for value in values) or args.lr <= 0 or min(values[1:]) < 0:
        raise ValueError('Invalid optimizer or selection settings')
    if args.task == 'domain' and not args.design:
        raise ValueError('--design is required for domain fine-tuning')
    if args.unfreeze_gnn and args.task != 'domain':
        raise ValueError('--unfreeze-gnn is available only for domain fine-tuning')
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use a CPU or CUDA device')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    original_hash = sha(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if sha(args.checkpoint) != original_hash:
        raise RuntimeError('Source checkpoint changed while loading')
    if checkpoint.get('model_version') != MODEL_VERSION:
        raise ValueError('Checkpoint must match the current GUALO model version')
    if checkpoint.get('settings', {}).get('target') != args.target:
        raise ValueError('Checkpoint target does not match --target')
    items, metadata, fingerprint = load_data(args, checkpoint)
    model = GUALO(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    parameters, hooks = configure(model, args.task, args.unfreeze_gnn)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.checkpoint, args.output / 'initial.pt')
    if sha(args.output / 'initial.pt') != original_hash or sha(args.checkpoint) != original_hash:
        raise RuntimeError('Source checkpoint changed while copying')
    initial = torch.load(args.output / 'initial.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(initial['model_state_dict'], strict=True)
    if not args.unfreeze_gnn:
        model.eval()
        with torch.no_grad():
            for item in items:
                item['embedding'] = model.gnn(graph_on_device(item, device)).detach().cpu()
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config = dict(settings=settings, parent_sha256=original_hash, dataset_fingerprint=fingerprint,
                  data=metadata, training_split='calibrate' if args.task == 'uq' else 'train',
                  selection_split='uq_validation' if args.task == 'uq' else 'select', test_used=False)
    atomic_json(args.output / 'config.json', config)
    validation_split = config['selection_split']
    baseline = evaluate(model, items, validation_split, args, device)
    atomic_json(args.output / 'baseline.json', baseline)
    best_score, best_epoch = selection_score(baseline, baseline, args), 0
    normalizations = {row['id']: row for row in initial.get('normalizations', [])}
    normalizations.update({item['id']: dict(id=item['id'], mean=item['mean'], std=item['std']) for item in items})

    def save_checkpoint(name, epoch, validation):
        payload = dict(model_version=MODEL_VERSION, training_version='gualo-finetune-v1',
                       model_config=initial['model_config'], model_state_dict=model.state_dict(),
                       settings=dict(target=args.target, seed=metadata['partition_seed']),
                       normalizations=list(normalizations.values()), epoch=epoch,
                       dataset_fingerprint=fingerprint, validation=validation,
                       finetune=config, optimizer_state_dict=optimizer.state_dict())
        atomic_save(args.output / name, payload)

    save_checkpoint('best.pt', 0, baseline)
    for epoch in range(1, args.epochs + 1):
        model.eval() if args.task == 'uq' else model.train()
        if not args.unfreeze_gnn:
            model.gnn.eval()
        rng = random.Random(args.seed + epoch)
        order = list(range(len(items)))
        rng.shuffle(order)
        total_loss, count = 0.0, 0
        for slot in order:
            item = items[slot]
            rows = item['splits'][config['training_split']].tolist()
            rng.shuffle(rows)
            batches = [rows[start:start + args.batch_size] for start in range(0, len(rows), args.batch_size)]
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2].extend(batches.pop())
            for batch, indices in enumerate(batches):
                seq = trim(item['seq'][indices]).to(device)
                y = item['y'][indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                embedding = embedding_for(model, item, device)
                if args.task == 'uq':
                    means, log_variances = monte_carlo(model, embedding, seq, args.mc_samples,
                                                       args.seed + epoch * 1000000 + slot * 10000 + batch)
                    loss = uncertainty_loss(means, log_variances, y, args.rank_weight)
                else:
                    mu, log_var = predict(model, embedding, seq)
                    loss = F.smooth_l1_loss(mu.flatten(), y)
                    loss = loss + args.rank_weight * pairwise_loss(mu, y, elite=True)
                    if args.task == 'domain':
                        loss = loss + nll_loss(mu, log_var, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite fine-tuning loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
                count += len(indices)
        validation = evaluate(model, items, validation_split, args, device)
        score = selection_score(validation, baseline, args)
        if score is not None and score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint('best.pt', epoch, validation)
        save_checkpoint('last.pt', epoch, validation)
        status = dict(epoch=epoch, task=args.task, train_loss=total_loss / count,
                      samples=count, best_epoch=best_epoch, validation=validation)
        atomic_json(args.output / f'epoch_{epoch:04d}.json', status)
        atomic_json(args.output / 'status.json', dict(phase='training', **status))
        print(json.dumps(status, allow_nan=False), flush=True)
    if args.task == 'uq':
        variance_keys = {f'transformer.{size}_mlp_head.1.{name}'
                         for size in ('sm', 'lg') for name in ('weight', 'bias')}
        for key, value in model.state_dict().items():
            original = initial['model_state_dict'][key]
            actual = value.detach().cpu()
            if not torch.equal(actual[0] if key in variance_keys else actual,
                               original[0] if key in variance_keys else original):
                raise RuntimeError(f'UQ fine-tuning changed a protected parameter: {key}')
    if sha(args.checkpoint) != original_hash:
        raise RuntimeError('Original checkpoint was modified externally')
    for hook in hooks:
        hook.remove()
    result = dict(phase='complete', task=args.task, epochs=args.epochs, best_epoch=best_epoch,
                  initial_sha256=original_hash, test_used=False)
    atomic_json(args.output / 'status.json', result)
    return result


if __name__ == '__main__':
    main()
