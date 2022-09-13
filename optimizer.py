import torch
import numpy as np


class ScheduledOptim:
    """https://github.com/ming024/FastSpeech2/blob/master/model/optimizer.py"""
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, hparams):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=[0.9, 0.98],
            eps=0.000000001,
            weight_decay=hparams.weight_decay,
        )
        self.n_warmup_steps = hparams.warm_up_step
        self.anneal_steps = hparams.anneal_steps
        self.anneal_rate = hparams.anneal_rate
        self.init_lr = np.power(hparams.d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
            