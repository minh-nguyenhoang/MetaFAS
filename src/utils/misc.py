import torch
from torch import inf

def init_random_seed(manual_seed = 1273):
    import torch
    import random
    import numpy as np
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("Use random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return 


class DummyLogger:
    def __init__(self,*args, **kwargs) -> None:
        pass
    def add_scalar(self,*args, **kwargs):
        pass
    def add_scalars(self,*args, **kwargs):
        pass
    def add_figure(self,*args, **kwargs):
        pass
    def add_image(self,*args, **kwargs):
        pass
    def add_images(self,*args, **kwargs):
        pass
    def add_text(self,*args, **kwargs):
        pass
    def add_audio(self,*args, **kwargs):
        pass

def count_parameters_in_MB(model):
    import numpy as np
    size = np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
    print(f'Model size: {size:<.3} MB')
    return size

class LimitedSizeList:
    '''List with limited size. No 'append' operation.
    '''
    def __init__(self, max_size, set_type = 'insert', init_value = None) -> None:
        self.init_value = init_value
        self.buffer = [init_value] * max_size
        self.set_type = set_type if set_type == 'insert' else '__setitem__'
        self.max_size = max_size

    def __len__(self):
        len = 0
        for ele in self.buffer:
            if ele is not None:
                len += 1
        return len
    
    def __iter__(self):
        return iter(self.buffer)

    def __next__(self):
        return next(self)
    def __getitem__(self, idx):
        return self.buffer[idx]

    def __setitem__(self, idx, value):
        from copy import deepcopy
        getattr(self.buffer, self.set_type, 'insert')(idx, deepcopy(value))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(-1)

    def set_max_size(self, new_size):
        assert isinstance(new_size, int) and new_size >0
        if new_size > self.max_size:
            self.buffer += [self.init_value] * (new_size - self.max_size)
            self.max_size = new_size
        else:
            self.buffer = self.buffer[:new_size]
            self.max_size = new_size

    def reset_list(self):
        self.buffer = [self.init_value] * self.max_size
        
        
        
def clip_grad_norm_(
        parameters, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    if norm_type == 'inf':
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        print(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, will set to zero before clipping .  ')

        for p in parameters:
            p_grad_=p.grad.detach()
            nan_idxs=torch.isnan(p_grad_)
            inf_idxs = torch.isinf(p_grad_)
            p_grad_[nan_idxs] = 0
            p_grad_[inf_idxs] = 0
        return clip_grad_norm_(parameters, max_norm, norm_type,error_if_nonfinite)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return total_norm



def clip_norm(
        parameters, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device

    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        print(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, will set to zero before clipping .  ')

        for p in parameters:
            p_grad_=p.detach()
            nan_idxs=torch.isnan(p_grad_)
            inf_idxs = torch.isinf(p_grad_)
            p_grad_[nan_idxs] = 0
            p_grad_[inf_idxs] = 0
        return clip_norm(parameters, max_norm, norm_type,error_if_nonfinite)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    grads = []
    for p in parameters:
        grads.append(p.detach().mul_(clip_coef_clamped.to(p.device)))
    return grads
