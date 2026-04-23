"""
DEIM: DETR with Improved Matching for Fast Convergence     
Copyright (c) 2024 The DEIM Authors. All Rights Reserved. 
"""

import math
from functools import partial
import matplotlib.pyplot as plt     
from ..extre_module.utils import plt_settings, TryExcept
from ..logger_module import get_logger
logger = get_logger(__name__)
  
def flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, current_iter, init_lr, min_lr):
    """ 
    Computes the learning rate using a warm-up, flat, and cosine decay schedule.  

    Args:
        total_iter (int): Total number of iterations.
        warmup_iter (int): Number of iterations for warm-up phase.
        flat_iter (int): Number of iterations for flat phase.
        no_aug_iter (int): Number of iterations for no-augmentation phase.
        current_iter (int): Current iteration.
        init_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        float: Calculated learning rate.
    """
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr
    elif current_iter >= total_iter - no_aug_iter:
        return min_lr
    else:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - flat_iter) /
                                           (total_iter - flat_iter - no_aug_iter)))
        return min_lr + (init_lr - min_lr) * cosine_decay


class FlatCosineLRScheduler:
    """
    Learning rate scheduler with warm-up, optional flat phase, and cosine decay following RTMDet.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance.
        lr_gamma (float): Scaling factor for the minimum learning rate.
        iter_per_epoch (int): Number of iterations per epoch.
        total_epochs (int): Total number of training epochs.
        warmup_epochs (int): Number of warm-up epochs.
        flat_epochs (int): Number of flat epochs (for flat-cosine scheduler).
        no_aug_epochs (int): Number of no-augmentation epochs.
        scheduler_type (str): Learning rate scheduler type (default "cosine").
    """

    def __init__(self, optimizer, lr_gamma, iter_per_epoch, total_epochs,
                 warmup_iter, flat_epochs, no_aug_epochs, scheduler_type="cosine", lr_scyedule_save_path=None):
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]

        total_iter = int(iter_per_epoch * total_epochs)
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)
        flat_iter = int(iter_per_epoch * flat_epochs)
        if flat_iter > total_iter:
            flat_iter = total_iter - warmup_iter

        logger.info(f"{self.base_lrs}, {self.min_lrs}, {total_iter}, {warmup_iter}, {flat_iter}, {no_aug_iter}")

        self.lr_func = partial(flat_cosine_schedule, total_iter, warmup_iter, flat_iter, no_aug_iter)

        for i, _ in enumerate(optimizer.param_groups):
            plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, self.base_lrs[i], self.min_lrs[i], lr_scyedule_save_path / f"lr_schedule_{i}.png")

    def step(self, current_iter, optimizer):
        """
        Updates the learning rate of the optimizer at the current iteration.

        Args:
            current_iter (int): Current iteration.
            optimizer (torch.optim.Optimizer): Optimizer instance.
        """
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = self.lr_func(current_iter, self.base_lrs[i], self.min_lrs[i])
        return optimizer

@TryExcept('WARNING ⚠️ plot_lr_schedule failed.')
@plt_settings()
def plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, init_lr, min_lr, save_path):
    is_four_stage = True
    iters = list(range(total_iter))
    if flat_iter == (total_iter - warmup_iter):
        is_four_stage = False
    lrs = [flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, i, init_lr, min_lr) for i in iters]

    plt.figure(figsize=(8, 5))
    plt.plot(iters, lrs, label='Learning Rate')
    plt.axvline(x=warmup_iter, color='r', linestyle='--', label='Warmup End')
    if is_four_stage:
        plt.axvline(x=flat_iter, color='g', linestyle='--', label='Flat End')
        plt.axvline(x=total_iter - no_aug_iter, color='b', linestyle='--', label='No Aug Start')
    else:
        plt.axvline(x=flat_iter + warmup_iter, color='g', linestyle='--', label='Flat End')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Flat Cosine Learning Rate Schedule')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close('all')

if __name__ == '__main__':
    total_iter = 10000
    warmup_iter = 1000
    flat_iter = 3000
    no_aug_iter = 500
    init_lr = 0.01
    min_lr = 0.0001

    plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, init_lr, min_lr)

