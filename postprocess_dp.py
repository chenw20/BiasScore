import torch
import numpy as np

import os
import sys
import argparse
import time
import json

import models
from utils import yaml_config_hook

from datasets import get_loaders
from global_config import update_global_config

import matplotlib.pyplot as plt


def DP(preds_y, gs):
    dp = torch.abs(torch.mean((preds_y[gs == 0] == 1).float()) -  torch.mean((preds_y[gs == 1] == 1).float()))
    return dp.item()

def DP_sign(preds_y, gs):
    dp = torch.mean((preds_y[gs == 0] == 1).float()) -  torch.mean((preds_y[gs == 1] == 1).float())
    return dp.item()

def Accuracy(preds_y, ys):
    return preds_y.eq(ys.view_as(preds_y)).sum().item() / preds_y.shape[0]

def compute_score(logits_y, pred_y, logits_g, pAeq0, pAeq1):
    eta = 2* torch.softmax(logits_y, 1).max(1, keepdim=True)[0] - 1
    
    pAeq0Gx = torch.softmax(logits_g, 1)[:,0]
    pAeq1Gx = torch.softmax(logits_g, 1)[:,1]

    f = (2 * (pred_y == 1).float() - 1)* (-pAeq1Gx / pAeq1 + pAeq0Gx / pAeq0)
    return f/eta.squeeze(1)

def get_pA(loader, dry_run=False):
    gs = []
    with torch.no_grad():
        for i, x, y, g in loader:
            gs.append(g)
            if dry_run:
                break
    gs = torch.cat(gs, dim=0)
    pAeq0 = torch.mean((gs == 0).float())
    pAeq1 = 1 - pAeq0
    return pAeq0, pAeq1

def get_ys_gs_logitsy_predys_logitsg(loader, model, dry_run=False):
    return model._get_ys_gs_logits_preds(loader, dry_run)


def run_expt(args, dry_run=False):
    seed = args['seed']
    n_epochs=args['n_epochs']
    args["log_dir"] = os.path.join(args["log_dir"], args["dataset"])
    args["log_dir"] = os.path.join(args["log_dir"], args["method"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    start_time = time.time()

    os.makedirs(args["log_dir"], exist_ok=True)
    checkpoint_file = os.path.join(
        args["log_dir"], 'seed_{}_{}.pt'.format(
            args["seed"], args["exp_id"]))
    checkpoint_file_y = os.path.join(
        args["log_dir"], 'seed_{}_{}_target.pt'.format(
            args["seed"], args["exp_id"]))
    checkpoint_file_g = os.path.join(
        args["log_dir"], 'seed_{}_{}_sensitive.pt'.format(
            args["seed"], args["exp_id"]))
    
    best_checkpoint_file = os.path.join(
        args["log_dir"],
        "seed_{}_{}.best.pt".format(args["seed"], args["exp_id"]),
    )
    best_checkpoint_file_y = os.path.join( 
        args["log_dir"],
        "seed_{}_{}.best_target.pt".format(args["seed"], args["exp_id"]),
    )
    best_checkpoint_file_g = os.path.join( 
        args["log_dir"],
        "seed_{}_{}.best_sensitive.pt".format(args["seed"], args["exp_id"]),
    )

    update_global_config(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("GPU will be used for testing.")
    else:
        print("WARNING: CPU will be used for testing.")


    loaders = get_loaders(args)

    model = {
        "erm": models.ERM,
        "dp_flip": models.ERM_DP_flip,
        "eo_flip": models.ERM_EO_flip,
    }[args["method"]](args, loaders['tr']).to(device)
    if args['method'] in ['dp_flip']:
        model.load(best_checkpoint_file_y, best_checkpoint_file_g)
    else:
        model.load(best_checkpoint_file)

    thresholds = np.linspace(0,50,5000)
    thresholds = np.append(thresholds, np.array(np.inf))
    ys, gs, logits_y, preds_y, logits_g, preds_g = get_ys_gs_logitsy_predys_logitsg(loaders['va'], model, dry_run)
    pAeq0, pAeq1 = get_pA(loaders['tr'], dry_run)
    scores_test = compute_score(logits_y, preds_y, logits_g, pAeq0, pAeq1)
    DP_list = []
    Accuracy_list = []
    s_list = []
    res = model.get_metrics(loaders['va'], ['acc', 'acc_g', 'dp'], dry_run)
    print('acc_y: {}'.format(res['acc']))
    print('acc_g: {}'.format(res['acc_g']))
    print('dp: {}'.format(res['dp']))

    DPs = DP_sign(preds_y, gs) 
    print('DP± = {}'.format(DPs))
    
    for s in np.flip(thresholds):
        cur_pred_y = preds_y.clone()

        if DPs > 0:
            loc = scores_test > s
            cur_pred_y_flip = torch.abs(cur_pred_y[loc] - 1) 
            cur_pred_y[loc] = cur_pred_y_flip
        else:
            loc = scores_test < -s
            cur_pred_y_flip = torch.abs(cur_pred_y[loc] - 1) 
            cur_pred_y[loc] = cur_pred_y_flip
        cur_DPs = DP_sign(cur_pred_y, gs) 
        if cur_DPs* DPs < 0:
            break
            
        DP_list.append(DP(cur_pred_y, gs))
        Accuracy_list.append(Accuracy(cur_pred_y, ys)) 
        s_list.append(s)
    DP_array = np.array(DP_list)
    Accuracy_array = np.array(Accuracy_list)
    s_array = np.array(s_list)

    cutoff = args['threshold']
    loc = DP_array <= cutoff
    s_array_ = s_array[loc]
    Accuracy_array_ = Accuracy_array[loc]
    DP_array_ = DP_array[loc]
    max_loc = np.argmax(Accuracy_array_)
    print('cutoff: {}, acc va: {}'.format(cutoff, Accuracy_array_[max_loc]))
    print('cutoff: {}, dp va: {}'.format(cutoff, DP_array_[max_loc]))
    score = s_array_[max_loc]
    

    res = model.get_metrics(loaders['te'], ['acc', 'acc_g', 'dp'], dry_run)
    print('acc_y: {}'.format(res['acc']))
    print('acc_g: {}'.format(res['acc_g']))
    print('dp: {}'.format(res['dp']))
    ys, gs, logits_y, preds_y, logits_g, preds_g = get_ys_gs_logitsy_predys_logitsg(loaders['te'], model, dry_run)
    scores_test = compute_score(logits_y, preds_y, logits_g, pAeq0, pAeq1)
    
    
    cur_pred_y = preds_y.clone()

    if DPs > 0:
        loc = scores_test > score
        cur_pred_y_flip = torch.abs(cur_pred_y[loc] - 1)
        cur_pred_y[loc] = cur_pred_y_flip
    else:
        loc = scores_test < -score
        cur_pred_y_flip = torch.abs(cur_pred_y[loc] - 1) 
        cur_pred_y[loc] = cur_pred_y_flip
    
    print('cutoff: {}, acc te: {}'.format(cutoff, Accuracy(cur_pred_y, ys)))
    print('cutoff: {}, dp te: {}'.format(cutoff, DP(cur_pred_y, gs)))
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fair score")
    config = yaml_config_hook("./config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args = vars(args)
    run_expt(args, False)
