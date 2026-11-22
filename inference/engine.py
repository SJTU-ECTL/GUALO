import argparse,sys,json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from .risk import RiskPredictor,extract_fused

class GUALOEngine:
 def __init__(self,checkpoint,device='cuda:0'):
  from model import GUALO
  self.ck=checkpoint;self.device=torch.device(device)
  self.model=GUALO(**self.ck['model_config']).to(self.device).eval();self.model.load_state_dict(self.ck['model_state_dict'],strict=True)
  self.model.requires_grad_(False);self.risk=RiskPredictor(self.ck)
 def predict(self,design,data,seeds=None,batch_size=128):
  from model import enable_dropout
  from integrity import stable_seed
  if design not in self.risk.settings['designs']:raise ValueError('No calibrated risk module for this design')
  target=self.risk.settings['target'];mean,std=self.risk.settings['designs'][design]['normalization'];seeds=self.risk.settings['mc_seeds'] if seeds is None else seeds
  if not seeds:raise ValueError('At least one MC seed is required')
  seq=data['opt_seq'].long();graph=Data(x=data['x'].long().to(self.device),edge_index=data['edge_index'].long().to(self.device),batch=torch.zeros(len(data['x']),dtype=torch.long,device=self.device))
  devices=[self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type=='cuda' else []
  modes=[(m,m.training) for m in self.model.modules()];outputs=[]
  try:
   with torch.random.fork_rng(devices=devices),torch.inference_mode():
    self.model.eval();emb=self.model.gnn(graph);enable_dropout(self.model)
    for seed in seeds:
     torch.manual_seed(stable_seed(f'{seed}:{target}:OpenCores__{design}'.encode(),2**31));chunks=[]
     for batch in seq.split(batch_size):
      batch=batch[:,:int(batch.ne(0).sum(1).max())].to(self.device);mus=[];lvs=[]
      for offset in range(0,30,4):
       count=min(4,30-offset);tokens=batch.repeat(count,1)
       raw=self.model.transformer.forward_encoded_graph(emb.expand(len(tokens),-1),tokens,False)
       mu,lv=self.model._split_prediction(raw);mus.append(mu.reshape(count,len(batch)).cpu());lvs.append(lv.reshape(count,len(batch)).cpu())
      mu=torch.cat(mus).double();lv=torch.cat(lvs).double()
      chunks.append(torch.stack([mu.mean(0),mu.var(0,unbiased=False),(lv*.5).exp().mean(0)]).numpy())
     outputs.append(np.concatenate(chunks,axis=1))
    values=np.stack(outputs);means=values[:,0]*std+mean;mu=means.mean(0);ep=np.sqrt(values[:,1].mean(0)*std**2+means.var(0));al=values[:,2].mean(0)*std
    fused,det=extract_fused(self.model,graph,seq,batch_size)
   result=self.risk.predict(design,seq.cpu().numpy(),fused,mu,ep,al,det*std+mean)
   result.update(epistemic=ep,aleatoric=al)
   return result
  finally:
   for module,mode in modes:module.training=mode
