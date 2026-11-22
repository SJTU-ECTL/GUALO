import numpy as np
import lightgbm as lgb
from .features import risk_features

class RiskPredictor:
 def __init__(self,checkpoint):
  self.settings=checkpoint['risk'];self.models={}
  self.shared={name:lgb.Booster(model_str=value) for name,value in self.settings.get('shared_models',{}).items()}
  for design,record in self.settings['designs'].items():
   self.models[design]={name:lgb.Booster(model_str=value) for name,value in record['models'].items()}
 def predict(self,design,tokens,fused,mu,epistemic,aleatoric,deterministic):
  if design not in self.models:raise ValueError('No risk module for this design')
  record=self.settings['designs'][design];mean,std=record['normalization']
  mu=np.asarray(mu);x=risk_features(tokens,fused,mu,epistemic,aleatoric,mean,std,deterministic)
  if x.shape[1]!=self.settings['feature_dim'] or not np.isfinite(x).all():raise ValueError('Invalid risk features')
  outputs={name:model.predict(x,num_threads=2) for name,model in self.models[design].items()}
  p=outputs.get('probability',np.full(len(mu),record['constant_probability']))
  outputs['probability']=p
  if self.shared:
   shared_x=np.column_stack([x,np.full(len(x),record['design_index'])]);blend=self.settings['shared_blend']
   for name,model in self.shared.items():outputs[name]=(1-blend)*outputs[name]+blend*model.predict(shared_x,num_threads=2)
   p=outputs['probability']
  calibration=self.settings['probability_calibration'];probability=np.interp(p,calibration['x'],calibration['y'])
  policy=record['policy'];risk=np.maximum(outputs[policy['model']],0)*std
  return dict(mean=mu.copy(),optimistic_probability=probability,raw_probability=p,optimism_penalty=risk,cost_score=mu+policy['kappa']*risk,enabled=policy['enabled'])

def extract_fused(model,graph,tokens,batch_size=128):
 import torch
 modes=[(m,m.training) for m in model.modules()];captured={};handles=[];features=[];means=[]
 def capture(name):
  def hook(module,args):captured[name]=args[0].detach()
  return hook
 try:
  model.eval()
  handles=[model.transformer.sm_mlp_head.register_forward_pre_hook(capture('sm')),model.transformer.lg_mlp_head.register_forward_pre_hook(capture('lg'))]
  device=next(model.parameters()).device
  with torch.inference_mode():
   g=model.gnn(graph)
   for seq in tokens.split(batch_size):
    seq=seq[:,:int(seq.ne(0).sum(1).max())].to(device)
    out=model.transformer.forward_encoded_graph(g.expand(len(seq),-1),seq,False)
    features.append(torch.cat([captured['sm'],captured['lg']],1).cpu().numpy());means.append(out[:,0].cpu().numpy())
  return np.concatenate(features),np.concatenate(means)
 finally:
  for h in handles:h.remove()
  for m,mode in modes:m.training=mode
