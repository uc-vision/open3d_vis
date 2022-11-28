from typing import Callable
import torch
import numpy as np

from structs.struct import Struct
import dataclasses


def to_numpy(points):
  if isinstance(points, torch.Tensor):
    return points.cpu().numpy()

  if isinstance(points, tuple):
    return tuple([to_numpy(p) for p in points])
  return points



def to_numpy_func(f, device='cpu'):
  def g(*args, **kwargs):
    args = [from_numpy(x, device) for x in args]
    kwargs = {k:from_numpy(x, device) for k, x in kwargs.items()}
    return to_numpy(f(*args, **kwargs))
  return g

def from_numpy(x, device='cpu'):
  if isinstance(x, np.ndarray):
    return torch.from_numpy(x).to(device)
  elif isinstance(x, torch.Tensor):
    return x.to(device)
  elif isinstance(x, tuple):
    return tuple([from_numpy(p, device=device) for p in x])
  elif isinstance(x, list):
    return [from_numpy(p, device=device) for p in x]
  elif isinstance(x, dict):
    return {k:from_numpy(p, device=device) for k, p in x.items()}    
  elif isinstance(x, Struct):
    return x._map(lambda t: from_numpy(t, device))
      
  elif isinstance(x, Callable):
    return to_numpy_func(x, device)

  elif dataclasses.is_dataclass(x):
    d = from_numpy(dataclasses.asdict(x), device=device)
    return x.__class__(**d)


  
  
  return x