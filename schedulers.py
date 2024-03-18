import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, power=0.9, last_epoch=-1) -> None:
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return self.base_lrs
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.power
            return [factor*lr for lr in self.base_lrs]


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [ratio * lr for lr in self.base_lrs]

    def get_lr_ratio(self):
        return self.get_warmup_ratio() if self.last_epoch < self.warmup_iter else self.get_main_ratio()

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ['linear', 'exp']
        alpha = self.last_epoch / self.warmup_iter

        return self.warmup_ratio + (1. - self.warmup_ratio) * alpha if self.warmup == 'linear' else self.warmup_ratio ** (1. - alpha)


class WarmupPolyLR(WarmupLR):
    def __init__(self, optimizer, power, max_iter, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.power = power
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter

        return (1 - alpha) ** self.power
    
# Copy from Detectron2
# def _get_warmup_factor_at_iter(
#     method: str, iter: int, warmup_iters: int, warmup_factor: float
# ) -> float:
#     """
#     Return the learning rate warmup factor at a specific iteration.
#     See :paper:`ImageNet in 1h` for more details.

#     Args:
#         method (str): warmup method; either "constant" or "linear".
#         iter (int): iteration at which to calculate the warmup factor.
#         warmup_iters (int): the number of warmup iterations.
#         warmup_factor (float): the base warmup factor (the meaning changes according
#             to the method used).

#     Returns:
#         float: the effective warmup factor at the given iteration.
#     """
#     if iter >= warmup_iters:
#         return 1.0

#     if method == "constant":
#         return warmup_factor
#     elif method == "linear":
#         alpha = iter / warmup_iters
#         return warmup_factor * (1 - alpha) + alpha
#     else:
#         raise ValueError("Unknown warmup method: {}".format(method))
    
# class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer: torch.optim.Optimizer,
#         max_iters: int,
#         warmup_factor: float = 0.001,
#         warmup_iters: int = 1000,
#         warmup_method: str = "linear",
#         last_epoch: int = -1,
#         power: float = 0.9,
#         constant_ending: float = 0.0,
#     ):
#         self.max_iters = max_iters
#         self.warmup_factor = warmup_factor
#         self.warmup_iters = warmup_iters
#         self.warmup_method = warmup_method
#         self.power = power
#         self.constant_ending = constant_ending
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self) -> List[float]:
#         warmup_factor = _get_warmup_factor_at_iter(
#             self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
#         )
#         if self.constant_ending > 0 and warmup_factor == 1.0:
#             # Constant ending lr.
#             if (
#                 math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
#                 < self.constant_ending
#             ):
#                 return [base_lr * self.constant_ending for base_lr in self.base_lrs]
#         return [
#             base_lr * warmup_factor * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
#             for base_lr in self.base_lrs
#         ]

#     def _compute_values(self) -> List[float]:
#         # The new interface
#         return self.get_lr()


class WarmupExpLR(WarmupLR):
    def __init__(self, optimizer, gamma, interval=1, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.gamma = gamma
        self.interval = interval
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        return self.gamma ** (real_iter // self.interval)


class WarmupCosineLR(WarmupLR):
    def __init__(self, optimizer, max_iter, eta_ratio=0, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.eta_ratio = eta_ratio
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        
        return self.eta_ratio + (1 - self.eta_ratio) * (1 + math.cos(math.pi * self.last_epoch / real_max_iter)) / 2



__all__ = ['polylr', 'warmuppolylr', 'warmupcosinelr', 'warmupsteplr']


def get_scheduler(scheduler_name: str, optimizer, max_iter: int, power: int, warmup_iter: int, warmup_ratio: float):
    assert scheduler_name in __all__, f"Unavailable scheduler name >> {scheduler_name}.\nAvailable schedulers: {__all__}"
    if scheduler_name == 'warmuppolylr':
        return WarmupPolyLR(optimizer, power, max_iter, warmup_iter, warmup_ratio, warmup='linear')
    elif scheduler_name == 'warmupcosinelr':
        return WarmupCosineLR(optimizer, max_iter, warmup_iter=warmup_iter, warmup_ratio=warmup_ratio)
    return PolyLR(optimizer, max_iter)