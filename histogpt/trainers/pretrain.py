""" 
PyTorch Lightning MIL Trainer
Author: Manuel Tran / Helmholtz Munich
"""

import torch
import torchmetrics

import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_metric_learning import losses


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Linear Warmup
    """
    def __init__(
        self, optimizer, warmup_steps, total_steps, min_lr,
        max_lr, eta_min=0, last_step=-1, verbose=False
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


class LightningCLS(pl.LightningModule):
    def __init__(self, cfg, model):
        """
        PyTorch Lightning Classifier
        """
        super().__init__()
        #self.save_hyperparameters()
        self.model = model

        self.betas = cfg.betas
        self.end_lr = cfg.end_lr
        self.max_lr = cfg.max_lr
        self.max_steps = cfg.max_steps
        self.min_lr = cfg.min_lr
        self.warm_steps = cfg.warm_steps
        self.wd = cfg.wd

        self.acc_train = torchmetrics.Accuracy(
            task=cfg.task, 
            num_classes=cfg.num_classes,
            average='micro',
        )
        self.acc_val = torchmetrics.Accuracy(
            task=cfg.task, 
            num_classes=cfg.num_classes,
            average='micro',
        )
        self.acc_test = torchmetrics.Accuracy(
            task=cfg.task, 
            num_classes=cfg.num_classes,
            average='micro',
        )

        self.f1_train = torchmetrics.F1Score(
            task=cfg.task, 
            num_classes=cfg.num_classes, 
            average='weighted',
        )
        self.f1_val = torchmetrics.F1Score(
            task=cfg.task, 
            num_classes=cfg.num_classes, 
            average='weighted',
        )
        self.f1_test = torchmetrics.F1Score(
            task=cfg.task, 
            num_classes=cfg.num_classes, 
            average='weighted',
        )

    def on_train_start(self):
        self.optimizers(use_pl_optimizer=False).param_groups[0]["lr"]
        self.optimizers().param_groups = (
            self.optimizers()._optimizer.param_groups
        )

    def configure_optimizers(self):
        """
        configuration of optimizers 
        """
        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.max_lr,
            betas=self.betas,
            eps=1e-08,
            weight_decay=self.wd,
        )

        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=self.warm_steps,
            total_steps=self.max_steps,
            max_lr = self.max_lr,
            min_lr=self.min_lr, 
            eta_min=self.end_lr,
        )
        return [optimizer], [scheduler]
    
    def forward(self, x): 
        logits = self.model(x)
        return logits

    def training_step(self, batch, indice):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.lr_schedulers().step()
       
        self.acc_train(preds, y)
        self.f1_train(preds, y)
        
        self.log("acc/train", self.acc_train, prog_bar=True)
        self.log("f1/train", self.f1_train, prog_bar=True)
        self.log("loss/train", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, indice):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.acc_val(preds, y)
        self.f1_val(preds, y)

        self.log("acc/val", self.acc_val, prog_bar=True)
        self.log("f1/val", self.f1_val, prog_bar=True)
        self.log("loss/val", loss, prog_bar=True)

    def test_step(self, batch, indice):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.acc_test(preds, y)
        self.f1_test(preds, y)

        self.log("acc/test", self.acc_test, prog_bar=True)
        self.log("f1/test", self.f1_test, prog_bar=True)
        self.log("loss/test", loss, prog_bar=True)
