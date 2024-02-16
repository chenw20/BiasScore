import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from networks import get_network


class ERM(torch.nn.Module):
    def __init__(self, hparams, dataloader):
        super().__init__()
        self.hparams = dict(hparams)
        dataset = dataloader.dataset
        self.dataset = dataset
        self.n_batches = len(dataloader)
        self.n_classes = len(set(dataset.y))
        self.n_groups = len(set(dataset.g))
        self.n_examples = len(dataset)
        self.last_epoch = 0
        self.best_selec_val = 0

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_no_reduction = nn.CrossEntropyLoss(reduction="none")
        self.init_network_()
        self.data_type="image"

    def init_network_(self):
        n_channels = {
            "celeba": 3, "adult": None, "compas": None,
        }[self.hparams["dataset"]]
        n_classes = {
            "celeba": 4, "adult": 4, "compas": 4,
        }[self.hparams["dataset"]]
        in_dim = {
            "celeba": None, "adult": 108, "compas": 9,
        }[self.hparams["dataset"]]
        if self.hparams['dataset'] == 'celeba':
            self.network = get_network(self.hparams["network"], n_channels, n_classes)
        elif self.hparams['dataset'] == 'adult':
            self.network = get_network(self.hparams["network"], in_dim, n_classes)
        elif self.hparams['dataset'] == 'compas':
            self.network = get_network(self.hparams["network"], in_dim, n_classes, hdim=16)
        else:
            raise NotImplementedError("Unknown dataset '{}'".format(self.hparams["dataset"]))

        self.optimizer = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }[self.hparams["optim"]](
            self.network.parameters(),
            **{
                    key: self.hparams[key]
                    for key in {
                        "sgd": ['lr', 'weight_decay', 'momentum'],
                        "adam": ['lr', 'weight_decay']
                    }[self.hparams["optim"]]
                    if key in self.hparams
            }
        )


        if self.hparams["scheduler"] == "exp":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.hparams["scheduler_gamma"]
            )

        elif self.hparams["scheduler"] == "linear":
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.01

            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0,
                end_factor=100 * self.hparams['lr'], total_iters=10
            )

        else:
            raise NotImplementedError("Unknown schduler '{}'".format(self.hparams["scheduler"]))

    def compute_loss_value_(self, i, x, y, g, epoch):
        loss = self.loss_ce(self.network(x), y)
        return loss 

    def _device(self):
        return next(self.parameters()).device

    def update(self, i, x, y, g, epoch):
        self.train()
        device = self._device()
        x, y, g = x.to(device), y.to(device), g.to(device)
        loss_value = self.compute_loss_value_(i, x, y, g, epoch)
        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()
        self.last_epoch = epoch

    def predict(self, x):
        return self.network(x)

    def _get_ys_gs_logits_preds(self, loader, dry_run=False):
        logits = []
        preds = []
        ys = []
        gs = []
        self.eval()
        with torch.no_grad():
            for i, x, y, g in loader:
                logit = self.predict(x.to(self._device())).detach().cpu()
                logit = torch.squeeze(logit, dim=1)
                if logit.ndim == 1:
                    pred = (logit > 0).long()
                else:
                    pred = logit.argmax(1)
                logits.append(logit)
                preds.append(pred)
                ys.append(y)
                gs.append(g)
                if dry_run:
                    break
        return [torch.cat(arr, dim=0) for arr in [ys, gs, logits, preds]]

    def get_metrics(self, loader, which_metrics, dry_run=False):
        for metric_name in which_metrics:
            assert(metric_name in ['acc', 'loss', 'acc_g', 'loss_g', 'dp', 'eo'])

        ys, gs, logits, preds = self._get_ys_gs_logits_preds(loader, dry_run=dry_run)

        res = {}
        if 'acc' in which_metrics:
            res['acc'] = preds.eq(ys).float().mean().item()
        if 'acc_g' in which_metrics:
            res['acc_g'] = preds.eq(gs).float().mean().item()
        if 'loss' in which_metrics:
            res['loss'] = self.loss_ce(logits, ys).item()
        if 'loss_g' in which_metrics:
            res['loss_g'] = self.loss_ce(logits, gs).item()
        if 'dp' in which_metrics:
            res['dp'] = (preds * gs).float().mean() / gs.float().mean() - (preds * (1 - gs)).float().mean() / (1 - gs).float().mean()
            res['dp'] = abs(res['dp'].item())
        if 'eo' in which_metrics:
            res['eo'] = 0.0
            for a in [ys, 1 - ys]:
                g1 = (a * gs).float()
                g2 = (a * (1 - gs)).float()
                res['eo'] = max(res['eo'], abs(((preds * g1).float().mean() / g1.mean() - (preds * g2).float().mean() / g2.mean()).item()))

        return res

    def accuracy(self, loader, dry_run=False):
        return self.get_metrics(loader, ['acc'], dry_run=dry_run)['acc']

    def load(self, fname):
        dicts = torch.load(fname, map_location=self._device())
        self.last_epoch = dicts["epoch"]
        self.load_state_dict(dicts["model"])
        self.optimizer.load_state_dict(dicts["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(dicts["scheduler"])

    def save(self, fname):
        lr_dict = None
        if self.lr_scheduler is not None:
            lr_dict = self.lr_scheduler.state_dict()
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": lr_dict,
                "epoch": self.last_epoch,
                "best_selec_val": self.best_selec_val,
            },
            fname,
        )


class ERM_EO_flip(ERM):
    def __init__(self, hparams, dataloader):
        super().__init__(hparams, dataloader)
        self.n_classes = len(set(self.dataset.y))* len(set(self.dataset.g))

    def compute_loss_value_(self, i, x, y, g, epoch):
        label = {
            "celeba": 2* g+ y,
            "adult": 2* g+ y,
            "compas": 2* g+ y,
        }[self.hparams["dataset"]]
        loss = self.loss_ce(self.network(x), label)
        return loss

    def get_metrics(self, loader, which_metrics, dry_run=False):
        for metric_name in which_metrics:
            assert(metric_name in ['acc', 'loss', 'acc_g', 'dp', 'eo'])

        ys, gs, logits, preds = self._get_ys_gs_logits_preds(loader, dry_run=dry_run)
        labels = {
            "celeba": 2* gs+ ys, "adult": 2* gs+ ys, "compas": 2* gs+ ys,
        }[self.hparams["dataset"]]
        preds_y = {
            "celeba": (preds % 2).long(), "adult": (preds % 2).long(), "compas": (preds % 2).long(),
        }[self.hparams["dataset"]]
        preds_g = {
            "celeba": (preds > 1.5).long(), "adult": (preds > 1.5).long(), "compas": (preds > 1.5).long(),
        }[self.hparams["dataset"]]

        res = {}
        if 'acc' in which_metrics:
            res['acc'] = preds_y.eq(ys).float().mean().item()
        if 'acc_g' in which_metrics:
            res['acc_g'] = preds_g.eq(gs).float().mean().item()
        if 'loss' in which_metrics:
            res['loss'] = self.loss_ce(logits, labels).item()
        if 'dp' in which_metrics:
            res['dp'] = (preds_y * gs).float().mean() / gs.float().mean() - (preds_y * (1 - gs)).float().mean() / (1 - gs).float().mean()
            res['dp'] = abs(res['dp'].item())
        if 'eo' in which_metrics:
            res['eo'] = 0.0
            for a in [ys, 1 - ys]:
                g1 = (a * gs).float()
                g2 = (a * (1 - gs)).float()
                res['eo'] = max(res['eo'], abs(((preds_y * g1).float().mean() / g1.mean() - (preds_y * g2).float().mean() / g2.mean()).item()))

        return res

    def accuracy_y(self, loader, dry_run=False):
        return self.get_metrics(loader, ['acc'], dry_run=dry_run)['acc']

    def accuracy_g(self, loader, dry_run=False):
        return self.get_metrics(loader, ['acc_g'], dry_run=dry_run)['acc_g']
    

class ERM_DP_flip(ERM):
    def __init__(self, hparams, dataloader):
        super().__init__(hparams, dataloader)
        self.n_classes = len(set(self.dataset.y))* len(set(self.dataset.g))
        self.init_network_()

    def init_network_(self):
        n_channels = {
            "celeba": 3, "adult": None, "compas": None,
        }[self.hparams["dataset"]]
        n_classes = {
            "celeba": 4, "adult": 4, "compas": 4,
        }[self.hparams["dataset"]]
        in_dim = {
            "celeba": None, "adult": 108, "compas": 9,
        }[self.hparams["dataset"]]
        n_groups = {
            "celeba": 2, "adult": 2, "compas": 2,
        }[self.hparams["dataset"]]
        if self.hparams['dataset'] == 'celeba':
            self.network_y = get_network(self.hparams["network"], n_channels, n_classes)
            self.network_g = get_network(self.hparams["network"], n_channels, n_groups)
        elif self.hparams['dataset'] == 'adult':
            self.network_y = get_network(self.hparams["network"], in_dim, n_classes)
            self.network_g = get_network(self.hparams["network"], in_dim, n_groups)
        elif self.hparams['dataset'] == 'compas':
            self.network_y = get_network(self.hparams["network"], in_dim, n_classes, hdim=16)
            self.network_g = get_network(self.hparams["network"], in_dim, n_classes, hdim=16)
        else:
            raise NotImplementedError("Unknown dataset '{}'".format(self.hparams["dataset"]))

        self.optimizer_y = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }[self.hparams["optim"]](
            self.network_y.parameters(),
            **{
                    key: self.hparams[key]
                    for key in {
                        "sgd": ['lr', 'weight_decay', 'momentum'],
                        "adam": ['lr', 'weight_decay']
                    }[self.hparams["optim"]]
                    if key in self.hparams
            }
        )

        self.optimizer_g = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }[self.hparams["optim"]](
            self.network_g.parameters(),
            **{
                    key: self.hparams[key]
                    for key in {
                        "sgd": ['lr', 'weight_decay', 'momentum'],
                        "adam": ['lr', 'weight_decay']
                    }[self.hparams["optim"]]
                    if key in self.hparams
            }
        )

        if self.hparams["scheduler"] == "exp":
            self.lr_scheduler_y = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer_y,
                gamma=self.hparams["scheduler_gamma"]
            )
            self.lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer_g,
                gamma=self.hparams["scheduler_gamma"]
            )

        elif self.hparams["scheduler"] == "linear":
            for param_group in self.optimizer_y.param_groups:
                param_group['lr'] = 0.01

            self.lr_scheduler_y = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_y, start_factor=1.0,
                end_factor=100 * self.hparams['lr'], total_iters=10
            )
            for param_group in self.optimizer_g.param_groups:
                param_group['lr'] = 0.01

            self.lr_scheduler_g = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_g, start_factor=1.0,
                end_factor=100 * self.hparams['lr'], total_iters=10
            )
        else:
            raise NotImplementedError("Unknown schduler '{}'".format(self.hparams["scheduler"]))

    def compute_loss_value_(self, i, x, y, g, epoch):
        loss_y = self.loss_ce(self.network_y(x), y)
        loss_g = self.loss_ce(self.network_g(x), g)
        return loss_y, loss_g


    def update(self, i, x, y, g, epoch):
        self.train()
        device = self._device()
        x, y, g = x.to(device), y.to(device), g.to(device)
        loss_value_y, loss_value_g = self.compute_loss_value_(i, x, y, g, epoch)
        self.optimizer_y.zero_grad()
        loss_value_y.backward()
        self.optimizer_y.step()
        
        self.optimizer_g.zero_grad()
        loss_value_g.backward()
        self.optimizer_g.step()
        self.last_epoch = epoch

    def predict_y(self, x):
        return self.network_y(x)
    
    def predict_g(self, x):
        return self.network_g(x)

    def _get_ys_gs_logits_preds(self, loader, dry_run=False):
        logits_y = []
        preds_y = []
        logits_g = []
        preds_g = []
        ys = []
        gs = []
        self.eval()
        with torch.no_grad():
            for i, x, y, g in loader:
                logit_y = self.predict_y(x.to(self._device())).detach().cpu()
                logit_y = torch.squeeze(logit_y, dim=1)
                if logit_y.ndim == 1:
                    pred_y = (logit_y > 0).long()
                else:
                    pred_y = logit_y.argmax(1)
                logits_y.append(logit_y)
                preds_y.append(pred_y)

                logit_g = self.predict_g(x.to(self._device())).detach().cpu()
                logit_g = torch.squeeze(logit_g, dim=1)
                if logit_g.ndim == 1:
                    pred_g = (logit_g > 0).long()
                else:
                    pred_g = logit_g.argmax(1)
                logits_g.append(logit_g)
                preds_g.append(pred_g)

                ys.append(y)
                gs.append(g)
                if dry_run:
                    break
        return [torch.cat(arr, dim=0) for arr in [ys, gs, logits_y, preds_y, logits_g, preds_g]]

    def get_metrics(self, loader, which_metrics, dry_run=False):
        for metric_name in which_metrics:
            assert(metric_name in ['acc', 'loss', 'acc_g', 'loss_g', 'dp', 'eo'])

        ys, gs, logits_y, preds_y, logits_g, preds_g = self._get_ys_gs_logits_preds(loader, dry_run=dry_run)

        res = {}
        if 'acc' in which_metrics:
            res['acc'] = preds_y.eq(ys).float().mean().item()
        if 'acc_g' in which_metrics:
            res['acc_g'] = preds_g.eq(gs).float().mean().item()
        if 'loss' in which_metrics:
            res['loss'] = self.loss_ce(logits_y, ys).item()
        if 'loss_g' in which_metrics:
            res['loss_g'] = self.loss_ce(logits_g, gs).item()
        if 'dp' in which_metrics:
            res['dp'] = (preds_y * gs).float().mean() / gs.float().mean() - (preds_y * (1 - gs)).float().mean() / (1 - gs).float().mean()
            res['dp'] = abs(res['dp'].item())
        if 'eo' in which_metrics:
            res['eo'] = 0.0
            for a in [ys, 1 - ys]:
                g1 = (a * gs).float()
                g2 = (a * (1 - gs)).float()
                res['eo'] = max(res['eo'], abs(((preds_y * g1).float().mean() / g1.mean() - (preds_y * g2).float().mean() / g2.mean()).item()))

        return res

    def accuracy_y(self, loader, dry_run=False):
        return self.get_metrics(loader, ['acc'], dry_run=dry_run)['acc']
    
    def accuracy_g(self, loader, dry_run=False):
        return self.get_metrics(loader, ['acc_g'], dry_run=dry_run)['acc_g']

    def load(self, fname_y, fname_g):
        dicts_y = torch.load(fname_y, map_location=self._device())
        dicts_g = torch.load(fname_g, map_location=self._device())
        self.last_epoch = dicts_y["epoch"]
        self.network_y.load_state_dict(dicts_y["model"])
        self.network_g.load_state_dict(dicts_g["model"])
        self.optimizer_y.load_state_dict(dicts_y["optimizer"])
        self.optimizer_g.load_state_dict(dicts_g["optimizer"])
        self.best_selec_val_y = dicts_y["best_selec_val"]
        self.best_selec_val_g = dicts_g["best_selec_val"]
        if self.lr_scheduler_y is not None:
            self.lr_scheduler_y.load_state_dict(dicts_y["scheduler"])
        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.load_state_dict(dicts_g["scheduler"])

    def save_y(self, fname):
        lr_dict = None
        if self.lr_scheduler_y is not None:
            lr_dict = self.lr_scheduler_y.state_dict()
        torch.save(
            {
                "model": self.network_y.state_dict(),
                "optimizer": self.optimizer_y.state_dict(),
                "scheduler": lr_dict,
                "epoch": self.last_epoch,
                "best_selec_val": self.best_selec_val_y,
            },
            fname,
        )
    
    def save_g(self, fname):
        lr_dict = None
        if self.lr_scheduler_g is not None:
            lr_dict = self.lr_scheduler_g.state_dict()
        torch.save(
            {
                "model": self.network_g.state_dict(),
                "optimizer": self.optimizer_g.state_dict(),
                "scheduler": lr_dict,
                "epoch": self.last_epoch,
                "best_selec_val": self.best_selec_val_g,
            },
            fname,
        )
