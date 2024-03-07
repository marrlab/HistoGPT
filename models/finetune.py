"""
Pytorch Lightning LM Trainer
© Manuel Tran / Helmholtz Munich
"""

import torch
import pytorch_lightning as pl

from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

#-------------------------------------------------------------------------------


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Linear Warmup
    """
    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        min_lr,
        max_lr,
        eta_min=0,
        last_step=-1,
        verbose=False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR,
              self).__init__(optimizer, last_step, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return [
                self.min_lr + (self.max_lr - self.min_lr) *
                (self._step_count) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            t = self._step_count - self.warmup_steps
            T = self.total_steps - self.warmup_steps
            return [
                self.eta_min + (self.max_lr - self.eta_min) *
                (1 + torch.cos(torch.tensor((t / T) * torch.pi)).item()) / 2
                for base_lr in self.base_lrs
            ]


#-------------------------------------------------------------------------------


class LanguageModeling(pl.LightningModule):
    """
    Lightning Trainer Class for Causal Language Modeling
    """
    def __init__(self, cfg, model):
        """
        :param cfg: hyperparameters for model optimization
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model

        self.betas = cfg.betas
        self.end_lr = cfg.end_lr
        self.max_lr = cfg.max_lr
        self.max_steps = cfg.max_steps
        self.min_lr = cfg.min_lr
        self.warm_steps = cfg.warm_steps
        self.wd = cfg.wd

    def on_train_start(self):
        self.optimizers(use_pl_optimizer=False).param_groups[0]["lr"]
        self.optimizers(
        ).param_groups = (self.optimizers()._optimizer.param_groups)

    def configure_optimizers(self):
        """
        configuration of optimizers 
        """
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = optim.AdamW(
            #params=self.parameters(),
            params=trainable_params,
            lr=self.max_lr,
            betas=self.betas,
            eps=1e-08,
            weight_decay=self.wd,
        )

        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=self.warm_steps,
            total_steps=self.max_steps,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            eta_min=self.end_lr,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, idx):
        """
        definition of training step
        """
        txt, img = batch
        #logits = self.model(txt[:, :-1], img.float()).logits
        logits = self.model(txt[:, :-1], None).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            txt[:, +1:].reshape(-1),
            ignore_index=-100
        )
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.lr_schedulers().step()
        return loss

    def validation_step(self, batch, idx):
        """
        definition of validation step
        """
        txt, img = batch
        #logits = self.model(txt[:, :-1], img.float()).logits
        logits = self.model(txt[:, :-1], None).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            txt[:, +1:].reshape(-1),
            ignore_index=-100
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, idx):
        """
        definition of test step
        """
        txt, img = batch
        #logits = self.model(txt[:, :-1], img.float()).logits
        logits = self.model(txt[:, :-1], None).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            txt[:, +1:].reshape(-1),
            ignore_index=-100
        )
        self.log("test_loss", loss, prog_bar=True)
        return loss
