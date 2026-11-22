import argparse
import json
import math
from pathlib import Path
import random
import time

import torch

from checkpointing import CheckpointRetention, atomic_json
from integrity import file_checksum
from data import CircuitDataset, graph_on_device, trim
from losses import joint_loss, nll_loss
from model import GUALO, MODEL_VERSION, enable_dropout


TRAINING_VERSION = 'gualo-joint-v1'


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def predict(model, embedding, seq, contrastive=False):
    output = model.transformer.forward_encoded_graph(embedding.expand(len(seq), -1), seq, contrastive)
    if not contrastive:
        return model._split_prediction(output)
    prediction, _, recipe = output
    mu, log_var = model._split_prediction(prediction)
    return mu, log_var, model.graph_proj_head(embedding), model.recipe_proj_head(recipe)


def correlation(x, y, rank=False):
    x, y = x.double().flatten(), y.double().flatten()
    if rank:
        def ranks(values):
            order = values.argsort(stable=True)
            _, inverse, counts = torch.unique_consecutive(values[order], return_inverse=True, return_counts=True)
            average = counts.cumsum(0).double() - (counts.double() - 1) / 2
            result = torch.empty_like(values)
            result[order] = average[inverse]
            return result
        x, y = ranks(x), ranks(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    return float((x @ y) / denominator) if denominator > 1e-12 else None


def average(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


@torch.no_grad()
def validate(model, dataset, device, mc_samples=30, do_uq=False, uq_circuits=12, uq_recipes=128):
    model.eval()
    nll_sum = ape_sum = count = 0
    designs, risks = [], []
    for slot, item in enumerate(dataset.items):
        embedding = model.gnn(graph_on_device(item, device))
        rows = item['splits']['validation']
        predictions = []
        for indices in rows.split(128):
            target = item['y'][indices].to(device)
            mu, log_var = predict(model, embedding, trim(item['seq'][indices]).to(device))
            predictions.append(mu.flatten().cpu())
            nll_sum += float(nll_loss(mu, log_var, target)) * len(indices)
        mu = torch.cat(predictions)
        raw_mu = mu * item['std'] + item['mean']
        error = (raw_mu - item['raw'][rows]).abs()
        ape = (error / item['raw'][rows]).sum().item()
        ape_sum += ape
        count += len(rows)
        k = max(1, math.ceil(len(rows) * .1))
        predicted_top = set(mu.argsort()[:k].tolist())
        true_top = set(item['y'][rows].argsort()[:k].tolist())
        designs.append(dict(id=item['id'],n=len(rows),mape=100*ape/len(rows),
                            rank=correlation(mu,item['y'][rows],rank=True),
                            top10_recall=len(predicted_top & true_top)/k))
        if do_uq and slot < uq_circuits:
            generator = torch.Generator().manual_seed(31415 + slot)
            indices = rows[torch.randperm(len(rows),generator=generator)[:uq_recipes]]
            seq = trim(item['seq'][indices]).to(device)
            enable_dropout(model)
            samples = [predict(model,embedding,seq) for _ in range(mc_samples)]
            mus = torch.stack([p[0].flatten() for p in samples]).cpu()
            lvs = torch.stack([p[1].flatten() for p in samples]).cpu()
            uncertainty = mus.std(0,unbiased=False) + .3 * (lvs*.5).exp().mean(0)
            error = (mus.mean(0) - item['y'][indices]).abs()
            k = max(1,math.ceil(len(indices)*.2))
            high = uncertainty.argsort()[-k:]
            worst = error.argsort()[-k:]
            risks.append(dict(id=item['id'],n=len(indices),
                              spearman=correlation(uncertainty,error,rank=True),
                              pearson=correlation(uncertainty,error),
                              top20_error_recall=len(set(high.tolist())&set(worst.tolist()))/k))
            model.eval()
    values = dict(mape=100*ape_sum/count,nll=nll_sum/count,
                  rank=average([d['rank'] for d in designs]),
                  uq=average([d['spearman'] for d in risks]))
    if not all(v is None or math.isfinite(v) for v in values.values()):
        raise RuntimeError('Validation produced nonfinite metrics')
    return dict(split='validation',n=count,values=values,per_circuit=designs,
                uncertainty=dict(mc_samples=mc_samples,per_circuit=risks))


def parser():
    p = argparse.ArgumentParser(description="Train GUALO for area or delay prediction.")
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--target',choices=['area','delay'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--lambda-recipe',type=float,default=.5)
    p.add_argument('--lambda-cross',type=float,default=.3)
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--margin',type=float,default=.05)
    p.add_argument('--negative-ratio',type=int,default=3)
    p.add_argument('--mc-samples',type=int,default=30)
    p.add_argument('--uq-every',type=int,default=5)
    p.add_argument('--snapshot-every',type=int,default=5)
    p.add_argument('--resume',type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.epochs,args.threads,args.top_k,args.negative_ratio,args.uq_every,args.snapshot_every) < 1 or args.batch_size < 8 or args.mc_samples < 2:
        raise ValueError('Positive counts, batch-size>=8 and mc-samples>=2 are required')
    if args.lr <= 0 or args.temperature <= 0 or min(args.lambda_recipe,args.lambda_cross) < 0 or not 0 <= args.margin <= 1:
        raise ValueError('Invalid loss or optimizer settings')
    if args.output.exists():
        raise FileExistsError('Use a NEW output directory, including when resuming; existing files are protected')
    device = torch.device(args.device)
    if device.type not in ('cpu','cuda'):
        raise ValueError('This entry point supports CPU or CUDA')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dataset = CircuitDataset(args.data,args.target,args.seed)
    model_config = dict(gnn_num_layers=5,graph_normalization='per_graph',mc_samples=args.mc_samples)
    model = GUALO(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=10,min_lr=1e-6)
    settings = {k:v for k,v in vars(args).items() if k not in ('data','output','resume','epochs','device','threads')}
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    start_epoch,step,parent = 1,0,None
    if args.resume:
        checkpoint = torch.load(args.resume,map_location='cpu',weights_only=True)
        if checkpoint.get('model_version') != MODEL_VERSION or checkpoint.get('training_version') != TRAINING_VERSION:
            raise ValueError('Checkpoint is not from this public training implementation')
        if not dataset.matches_fingerprint(checkpoint['dataset_fingerprint']) or checkpoint['settings'] != settings or checkpoint['model_config'] != model_config:
            raise ValueError('Resume requires identical dataset, target and training settings')
        start_epoch,step = checkpoint['epoch']+1,checkpoint['step']
        if start_epoch > args.epochs:
            raise ValueError('--epochs must exceed the resume checkpoint epoch')
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        torch.set_rng_state(checkpoint['rng_cpu'])
        random.setstate(checkpoint['rng_python'])
        if cuda_devices and checkpoint['rng_cuda'] is not None:
            torch.cuda.set_rng_state(checkpoint['rng_cuda'],device)
        parent = dict(file=str(args.resume.resolve()),checksum=file_checksum(args.resume))
    args.output.mkdir(parents=True,exist_ok=False)
    source_root = Path(__file__).resolve().parent
    source_files = [source_root/n for n in ('train.py','data.py','losses.py','checkpointing.py','integrity.py')] + sorted((source_root/'model').glob('*.py'))
    config = dict(model_version=MODEL_VERSION,training_version=TRAINING_VERSION,model_config=model_config,
                  settings=settings,epochs=args.epochs,device=str(device),dataset_fingerprint=dataset.fingerprint,
                  parent=parent,source_checksums={str(p.relative_to(source_root)):file_checksum(p) for p in source_files})
    atomic_json(args.output/'config.json',config)
    atomic_json(args.output/'dataset.json',dataset.metadata)
    retention = CheckpointRetention(args.output,dict(target=args.target,split='validation',
        dataset_fingerprint=dataset.fingerprint,model_version=MODEL_VERSION,
        evaluation='per_graph BN; deterministic QoR; fixed subset MC risk',
        scope='Best in this run; test recipes never select checkpoints'),every=args.snapshot_every)
    started = time.monotonic()
    for epoch in range(start_epoch,args.epochs+1):
        model.train()
        order = list(range(len(dataset.items)))
        rng = random.Random(args.seed+epoch)
        rng.shuffle(order)
        sums,count = [0.0]*4,0
        for slot in order:
            item = dataset.items[slot]
            graph = graph_on_device(item,device)
            rows = item['splits']['train'].tolist()
            rng.shuffle(rows)
            chunks = [rows[i:i+args.batch_size] for i in range(0,len(rows),args.batch_size)]
            if len(chunks)>1 and len(chunks[-1])<8:
                chunks[-2].extend(chunks.pop())
            for indices in chunks:
                seq = trim(item['seq'][indices]).to(device)
                target = item['y'][indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                mu,lv,gp,rp = predict(model,model.gnn(graph),seq,contrastive=True)
                losses = joint_loss(mu,lv,target,gp,rp,lambda_recipe=args.lambda_recipe,
                    lambda_cross=args.lambda_cross,temperature=args.temperature,top_k=args.top_k,
                    margin=args.margin,negative_ratio=args.negative_ratio)
                if not all(torch.isfinite(loss).all() for loss in losses):
                    raise RuntimeError(f'Nonfinite training loss at epoch {epoch}, circuit {item["id"]}')
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step()
                sums = [total+float(loss.detach())*len(indices) for total,loss in zip(sums,losses)]
                count += len(indices)
                step += 1
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(2026)
            validation = validate(model,dataset,device,args.mc_samples,
                do_uq=(epoch==start_epoch or epoch%args.uq_every==0 or epoch==args.epochs))
        scheduler.step(validation['values']['nll'])
        metrics = dict(epoch=epoch,step=step,train_losses=[v/count for v in sums],validation=validation,
                       elapsed_seconds=time.monotonic()-started,lr=optimizer.param_groups[0]['lr'])
        checkpoint = dict(model_version=MODEL_VERSION,training_version=TRAINING_VERSION,model_config=model_config,
            epoch=epoch,step=step,settings=settings,dataset_fingerprint=dataset.fingerprint,
            normalizations=[dict(id=i['id'],mean=i['mean'],std=i['std']) for i in dataset.items],
            model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(device) if cuda_devices else None,rng_python=random.getstate(),
            config=config,validation=validation)
        retention.consider(checkpoint,epoch,validation['values'],force=epoch==args.epochs)
        atomic_save(args.output/'last.pt',checkpoint)
        atomic_json(args.output/f'metrics_epoch_{epoch:04d}.json',metrics)
        with (args.output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(epoch=epoch,train_losses=metrics['train_losses'],validation=validation['values']))+'\n')
        status = dict(phase='epoch_complete',epoch=epoch,epochs=args.epochs,values=validation['values'])
        atomic_json(args.output/'status.json',status)
        print(json.dumps(status),flush=True)
    result = dict(phase='complete',epochs=args.epochs,step=step,best=retention.state['best'],test_used=False)
    atomic_json(args.output/'status.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    main()
