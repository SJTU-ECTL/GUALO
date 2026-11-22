import json
import math
import os
from pathlib import Path
import re
import shutil

from integrity import LEGACY_CHECKSUM, file_checksum

POLICY_VERSION = 'validation-multi-objective-v1'
RULES = {
    'mape': ('min', 'Validation MAPE; smaller is better.'),
    'nll': ('min', 'Validation Gaussian NLL; not a substitute for error detection.'),
    'rank': ('max', 'Macro per-circuit Spearman correlation; larger is better.'),
    'uq': ('max', 'Macro uncertainty/error Spearman on the fixed MC validation subset.'),
}


def atomic_json(path, data):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def default_writer(path, payload):
    import torch
    torch.save(payload, path)


def atomic_alias(source, destination):

    destination = Path(destination); tmp = destination.with_name(destination.name + '.next')
    if tmp.exists(): tmp.unlink()
    try: os.link(source, tmp)
    except OSError: shutil.copy2(source, tmp)
    tmp.replace(destination)


class CheckpointRetention:
    def __init__(self, out, context, every=5, writer=None):
        if every < 1: raise ValueError('Snapshot interval must be positive')
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out/'checkpoint_selection.json'
        self.directory = self.out/'retained_checkpoints'; self.directory.mkdir(exist_ok=True)
        self.writer = writer or default_writer
        self.context, self.every = context, every
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['policy_version'] != POLICY_VERSION or self.state['context'] != context or self.state['every'] != every:
                raise ValueError('Resume checkpoint policy/context differs from this run')

            for key, record in self.state['best'].items():
                source = self.out/record['file']
                legacy = 'checksum' not in record
                expected = record[LEGACY_CHECKSUM if legacy else 'checksum']
                if file_checksum(source, legacy=legacy) != expected: raise ValueError('Retained checkpoint was modified')
                atomic_alias(source, self.out/f'best_{key}.pt')
        else:
            if any((self.out/f'best_{key}.pt').exists() for key in RULES):
                raise FileExistsError('Existing weights are unmanaged; choose a new output directory')
            self.state = dict(policy_version=POLICY_VERSION, context=context, every=every,
                rules={k:dict(direction=v[0],meaning=v[1]) for k,v in RULES.items()},
                best={}, retained=[], evaluations=[])
            atomic_json(self.path, self.state)

    def consider(self, payload, epoch, values, *, split='validation', tag='train', eligible=None, force=False):
        if split not in ('validation', 'select'):
            raise ValueError('Test/calibration/check data cannot select checkpoints')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', tag): raise ValueError('Invalid checkpoint tag')
        if epoch < 0: raise ValueError('Invalid epoch')
        key=f'{epoch}:{tag}'
        if any(row['key']==key for row in self.state['evaluations']):
            raise ValueError('Duplicate epoch/tag would replace retained evidence')
        eligible = eligible or {}
        valid = {k:float(v) for k,v in values.items() if v is not None and math.isfinite(float(v))}
        improved=[]
        for name,(direction,_) in RULES.items():
            if name not in valid or not eligible.get(name, True): continue
            old=self.state['best'].get(name)
            if old is None or (valid[name] < old['value'] if direction=='min' else valid[name] > old['value']):
                improved.append(name)
        periodic=epoch % self.every == 0
        record=dict(key=key,epoch=epoch,tag=tag,split=split,values=valid,eligible=eligible,improved=improved,periodic=periodic)
        if improved or periodic or force:
            destination=self.directory/f'epoch_{epoch:06d}_{tag}.pt'
            if destination.exists(): raise FileExistsError(str(destination))
            saved=dict(payload,retention=dict(policy_version=POLICY_VERSION,context=self.context,**record))
            tmp=destination.with_suffix('.pt.tmp')
            self.writer(tmp,saved);tmp.replace(destination);destination.chmod(0o444)
            record.update(file=str(destination.relative_to(self.out)),checksum=file_checksum(destination))
            self.state['retained'].append(dict(record))
            for name in improved:
                self.state['best'][name]=dict(epoch=epoch,tag=tag,value=valid[name],file=record['file'],checksum=record['checksum'])
        self.state['evaluations'].append(record)
        atomic_json(self.path,self.state)
        for name in improved:atomic_alias(self.out/record['file'],self.out/f'best_{name}.pt')
        return improved
