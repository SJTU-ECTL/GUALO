import numpy as np

def recipe_features(tokens):
 tokens=np.asarray(tokens,dtype=int);n=len(tokens)
 counts=np.column_stack([(tokens==i).sum(1) for i in range(1,14)])
 length=(tokens!=0).sum(1)
 bigram=np.zeros((n,13*13),dtype=np.float32)
 for row,seq in enumerate(tokens):
  seq=seq[seq!=0]
  if len(seq)>1:np.add.at(bigram[row],(seq[:-1]-1)*13+seq[1:]-1,1)
 ends=np.column_stack([tokens[:,0],tokens[np.arange(n),length-1]])
 return np.column_stack([length,counts,counts/length[:,None],bigram,ends]).astype(np.float32)

def risk_features(tokens,fused,mu,epistemic,aleatoric,mean,std,deterministic):
 scalar=np.column_stack([(mu-mean)/std,epistemic/std,aleatoric/std,(deterministic-mu)/std])
 return np.column_stack([recipe_features(tokens),fused,scalar]).astype(np.float32)
