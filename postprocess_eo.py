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

def Accuracy_y(pred, target, flag_y=False):
    if flag_y:
        pred_y = pred
    else:
        pred_y = (pred % 2).long()
    target_y = (target % 2).long()
    return torch.sum(pred_y.eq(target_y.view_as(pred_y)), -1) / target_y.shape[-1]

def Accuracy_a(pred, target):
    pred_a = (pred > 1.5).long()  
    target_a = (target > 1.5).long()
    return pred_a.eq(target_a.view_as(pred_a)).sum().item() / pred_a.shape[0]

def EO(pred, target, flag_y=False):
    if flag_y:
        pred_y = pred
    else:
        pred_y = (pred % 2).long()
    EO_Y0 = torch.abs(torch.mean(((target == 0).float() + (pred_y == 1).float() > 1.5).float(), -1) / torch.mean((target == 0).float(), -1) - torch.mean(((target == 2).float() + (pred_y == 1).float() > 1.5).float(), -1) / torch.mean((target == 2).float(), -1))
    EO_Y1 = torch.abs(torch.mean(((target == 1).float() + (pred_y == 1).float() > 1.5).float(), -1) / torch.mean((target == 1).float(), -1) - torch.mean(((target == 3).float() + (pred_y == 1).float() > 1.5).float(), -1) / torch.mean((target == 3).float(), -1))
    return EO_Y0, EO_Y1

def EO_sign(pred, target, flag_y=False):
    if flag_y:
        pred_y = pred
    else:
        pred_y = (pred % 2).long()
    EO_Y0 = torch.mean(((target == 0).float() + (pred_y == 1).float() > 1.5).float(), -1) / torch.mean((target == 0).float(), -1) - torch.mean(((target == 2).float() + (pred_y == 1).float() > 1.5).float(), -1) / torch.mean((target == 2).float(), -1)
    EO_Y1 = torch.mean(((target == 1).float() + (pred_y == 1).float() > 1.5).float(), -1) / torch.mean((target == 1).float(), -1) - torch.mean(((target == 3).float() + (pred_y == 1).float() > 1.5).float(), -1) / torch.mean((target == 3).float(), -1)
    return EO_Y0, EO_Y1

def get_ys_gs_logits_preds(model, loader, dry_run=False):
    return model._get_ys_gs_logits_preds(loader, dry_run)

def get_pAY(loader, dry_run=False):
    labels = []
    with torch.no_grad():
        for i, x, y, g in loader:
            labels.append(2* g+ y)
            if dry_run:
                break
    labels = torch.cat(labels, dim=0)
    pA0Y0 = torch.mean((labels == 0).float())
    pA0Y1 = torch.mean((labels == 1).float())
    pA1Y0 = torch.mean((labels == 2).float())
    pA1Y1 = torch.mean((labels == 3).float())
    return pA0Y0, pA0Y1, pA1Y0, pA1Y1

def compute_score(logits, pA0Y0, pA0Y1, pA1Y0, pA1Y1):
    pred = logits.max(1, keepdim=True)[1]
    pred_y = (pred % 2).long().squeeze(1)

    prob = torch.softmax(logits, 1)
    pA0Y0Gx = prob[:,0]
    pA1Y0Gx = prob[:,2]
    pA0Y1Gx = prob[:,1]
    pA1Y1Gx = prob[:,3]

    eta = torch.zeros_like(pred_y).float()
    eta[pred_y == 1] = (2* (pA0Y1Gx + pA1Y1Gx) - 1)[pred_y == 1]
    eta[pred_y == 0] = (2* (pA0Y0Gx + pA1Y0Gx) - 1)[pred_y == 0]
    
    f1 = (2 * (pred_y == 1).float() - 1) * (pA0Y0Gx / pA0Y0 - pA1Y0Gx / pA1Y0)
    f2 = (2 * (pred_y == 1).float() - 1) * (pA0Y1Gx / pA0Y1 - pA1Y1Gx / pA1Y1)

    return f1, f2, eta

def run_expt(args, dry_run=False):
    seed = args['seed']
    n_epochs=args['n_epochs']
    bs_test = args['bs_test']
    threshold = args['threshold']
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


    ys, gs, logits, preds = get_ys_gs_logits_preds(model, loaders['va'], dry_run)
    ind = torch.randperm(logits.size()[0]) 
    logits = logits[ind] 
    ys = ys[ind]
    gs = gs[ind]
    preds = preds[ind]
    
    pA0Y0, pA0Y1, pA1Y0, pA1Y1 = get_pAY(loaders['tr'], dry_run)
    f1, f2, eta = compute_score(logits, pA0Y0, pA0Y1, pA1Y0, pA1Y1)
    score1 = f1 / eta
    score2 = f2 / eta
    EO_Y0_sign, EO_Y1_sign = EO_sign(preds, 2* gs + ys)

    preds_y = (preds % 2).long()
    
    M = max(len(score1), 5000)
    score1_subset = score1[:M]
    score2_subset = score2[:M]
    if EO_Y0_sign > 0 and EO_Y1_sign > 0:
        loc1 = (score1_subset > 0)
        loc2 = (score2_subset > 0)
        loc = (loc1.float() + loc2.float() >1.5)
    elif EO_Y0_sign <= 0 and EO_Y1_sign <= 0:
        loc1 = (score1_subset < 0)
        loc2 = (score2_subset < 0)
        loc = (loc1.float() + loc2.float() >1.5)
    elif EO_Y0_sign > 0 and EO_Y1_sign <= 0:
        loc1 = (score1_subset > 0)
        loc2 = (score2_subset < 0)
        loc = (loc1.float() + loc2.float() >1.5)
    else:
        loc1 = (score1_subset < 0)
        loc2 = (score2_subset > 0)
        loc = (loc1.float() + loc2.float() >1.5)
    
    N = len(score1_subset[loc])
    score1_mat = score1_subset[loc].unsqueeze(1) - score1_subset[loc].unsqueeze(0) 
    score2_mat = score2_subset[loc].unsqueeze(1) - score2_subset[loc].unsqueeze(0) 

    slope_mat = score2_mat / score1_mat 
    anchor_x = score1_subset[loc].unsqueeze(1).tile(1, N) 
    anchor_y = score2_subset[loc].unsqueeze(1).tile(1, N) 
    slope_mat = slope_mat[torch.tril_indices(N,N,-1).unbind()] 
    anchor_x = anchor_x[torch.tril_indices(N,N,-1).unbind()] 
    anchor_y = anchor_y[torch.tril_indices(N,N,-1).unbind()] 

    iteration = len(slope_mat) // bs_test + 1
    print('total iterations: {}'.format(iteration))
    print('val size: {}'.format(ys.shape[0]))

    max_acc = 0.
    eo0 = 10000.
    eo1 = 10000.

    start = time.time()
    
    for i in range(iteration):
        cur_pred_y = preds_y.clone() 
        if i != iteration - 1:
            loc = score2.unsqueeze(0) < anchor_y[i* bs_test: (i+1)* bs_test].unsqueeze(1) \
                + slope_mat[i* bs_test: (i+1)* bs_test].unsqueeze(1)* (score1.unsqueeze(0) \
                            - anchor_x[i* bs_test: (i+1)* bs_test].unsqueeze(1)) 
        else:
            loc = score2.unsqueeze(0) < anchor_y[i* bs_test:].unsqueeze(1) \
                + slope_mat[i* bs_test:].unsqueeze(1)* (score1.unsqueeze(0) \
                                - anchor_x[i* bs_test:].unsqueeze(1)) 
        loc[torch.mean(loc.float(), 1) > 0.5, :] = (1- loc[torch.mean(loc.float(), 1) > 0.5, :].float()).bool()
    
        cur_pred_y = preds_y.clone() 

        cur_pred_y = cur_pred_y.unsqueeze(0).tile(loc.shape[0], 1) 
        cur_pred_y_flip = torch.abs(cur_pred_y[loc] - 1) 
        cur_pred_y[loc] = cur_pred_y_flip  
  
        EO0, EO1 = EO(cur_pred_y, (2* gs+ ys).unsqueeze(0).tile(loc.shape[0], 1), True) 
        acc = Accuracy_y(cur_pred_y, (2* gs+ ys).unsqueeze(0).tile(loc.shape[0], 1), True)

        loc1 = (EO0 < threshold).float()
        loc2 = (EO1 < threshold).float()
        loc = (loc1 + loc2 > 1.5)
        if torch.sum(loc.float()) > 0:
            EO0 = EO0[loc]
            EO1 = EO1[loc]
            acc = acc[loc]
            opt_acc = acc[torch.argmax(acc)]
            opt_eo0 = EO0[torch.argmax(acc)]
            opt_eo1 = EO1[torch.argmax(acc)]
            if opt_acc > max_acc:
                max_acc = opt_acc
                eo0 = opt_eo0
                eo1 = opt_eo1
                slope = slope_mat[i* bs_test: (i+1)* bs_test][loc][torch.argmax(acc)].item()
                cur_anchor_x = anchor_x[i* bs_test: (i+1)* bs_test][loc][torch.argmax(acc)].item()
                cur_anchor_y = anchor_y[i* bs_test: (i+1)* bs_test][loc][torch.argmax(acc)].item()

        end = time.time()
        print('iteration: {}, time: {}'.format(i+1, end-start))
        print('thereshold: {}, acc:{}, eo0: {}, eo1: {}'.format(threshold, max_acc, eo0, eo1))
        start = time.time()
    
    res = model.get_metrics(loaders['va'], ['acc', 'acc_g', 'eo'], dry_run)
    print('acc_y: {}'.format(res['acc']))
    print('acc_g: {}'.format(res['acc_g']))
    print('eo: {}'.format(res['eo']))
    print('va, thereshold: {}, acc:{}, eo0: {}, eo1: {}'.format(threshold, max_acc, eo0, eo1))

    ys, gs, logits, preds = get_ys_gs_logits_preds(model, loaders['te'], dry_run)
    f1, f2, eta = compute_score(logits, pA0Y0, pA0Y1, pA1Y0, pA1Y1)
    score1 = f1 / eta
    score2 = f2 / eta
    preds_y = (preds % 2).long()
    
    res = model.get_metrics(loaders['te'], ['acc', 'acc_g', 'eo'], dry_run)
    print('acc_y: {}'.format(res['acc']))
    print('acc_g: {}'.format(res['acc_g']))
    print('eo: {}'.format(res['eo']))

    if slope != None:
        loc = score2 < cur_anchor_y + slope * (score1 - cur_anchor_x)
        if torch.mean(loc.float()) > 0.5:
            loc = (1- loc.float()).bool()
        cur_pred_y = preds_y.clone() 
        cur_pred_y_flip = torch.abs(cur_pred_y[loc] - 1)
        cur_pred_y[loc] = cur_pred_y_flip  
        EO0, EO1 = EO(cur_pred_y, 2* gs+ ys, True) 
        acc = Accuracy_y(cur_pred_y, 2* gs+ ys, True) 
        print('te, thereshold: 0.1, acc:{}, eo0: {}, eo1: {}'.format(acc, EO0, EO1))

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fair score")
    config = yaml_config_hook("./config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args = vars(args)
    bs_test = 10000
    threshold = 0.01
    run_expt(args, False)
