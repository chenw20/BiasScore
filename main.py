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

class Tee:
    def __init__(self, fname, stream, mode="a+"):
        self.stream = stream
        self.file = open(fname, mode)

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def print(self, message):
        self.write(message + "\n")

    def flush(self):
        self.stream.flush()
        self.file.flush()

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
    stdout = Tee(os.path.join(
        args["log_dir"], 'seed_{}_{}.out'.format(
            args["seed"], args["exp_id"])), sys.stdout)
    stderr = Tee(os.path.join(
        args["log_dir"], 'seed_{}_{}.err'.format(
            args["seed"], args["exp_id"])), sys.stderr)
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
        print("GPU will be used for training.")
    else:
        print("WARNING: CPU will be used for training.")

    loaders = get_loaders(args)

    model = {
        "erm": models.ERM,
        "dp_flip": models.ERM_DP_flip,
        "eo_flip": models.ERM_EO_flip,
    }[args["method"]](args, loaders['tr']).to(device)

    last_epoch = 0
    if args['method'] in ['erm', 'eo_flip']:
        best_selec_val = float('-inf')
        if os.path.exists(checkpoint_file):
            model.load(checkpoint_file)
            last_epoch = model.last_epoch
            best_selec_val = model.best_selec_val
    elif args['method'] in ["dp_flip"]:
        best_selec_val_y = float('-inf')
        best_selec_val_g = float('-inf')
        if os.path.exists(checkpoint_file_y) and os.path.exists(checkpoint_file_g):
            model.load(checkpoint_file_y, checkpoint_file_g)
            last_epoch = model.last_epoch
            best_selec_val_y = model.best_selec_val_y
            best_selec_val_g = model.best_selec_val_g
    else:
        raise NotImplementedError("Unknown method. Please choose specify one method from dp_flip and eo_flip")

    selec_func = {
        "erm": lambda m: m['acc'],
        "eo_flip": lambda m: m['acc'],
        "dp_flip": lambda m: (m['acc'], m['acc_g'])
    }[args['method']]

    for epoch in range(last_epoch, n_epochs):
        for i, x, y, g in loaders['tr']:
            model.update(i, x, y, g, epoch)
            if dry_run:
                break

        result = {"args": args, "epoch": epoch, "time": time.time() - start_time}

        for loader_name, loader in loaders.items():
            if loader_name in ["tr", "va", "te"]:
                if args['method'] == 'erm':
                    metrics = model.get_metrics(loader, ['loss', 'acc', 'dp', 'eo'], dry_run=dry_run)
                elif args['method'] == 'eo_flip':
                        metrics = model.get_metrics(loader, ['loss', 'acc', 'acc_g', 'dp', 'eo'], dry_run=dry_run)
                elif args['method'] == 'dp_flip':
                    metrics = model.get_metrics(loader, ['loss', 'acc', 'loss_g', 'acc_g', 'dp', 'eo'], dry_run=dry_run)
                else:
                    raise NotImplementedError("Unknown method. Please choose specify one method from dp_flip and eo_flip")
                result['metrics_' + loader_name] = metrics

        stdout.print(json.dumps(result))

        if args['method'] in ['erm', 'eo_flip']:
            selec_value = selec_func(result['metrics_va'])

            if selec_value >= best_selec_val:
                model.best_selec_val = selec_value
                best_selec_val = selec_value
                model.save(best_checkpoint_file)
            model.save(checkpoint_file)
            model.lr_scheduler.step()
        elif args['method'] in ['dp_flip']:
            selec_value_y, selec_value_g = selec_func(result['metrics_va'])

            if selec_value_y >= best_selec_val_y:
                model.best_selec_val_y = selec_value_y
                best_selec_val_y = selec_value_y
                model.save_y(best_checkpoint_file_y)
            model.save_y(checkpoint_file_y)
            model.lr_scheduler_y.step()

            if selec_value_g >= best_selec_val_g:
                model.best_selec_val_g = selec_value_g
                best_selec_val_g = selec_value_g
                model.save_g(best_checkpoint_file_g)
            model.save_g(checkpoint_file_g)
            model.lr_scheduler_g.step()
        else:
            raise NotImplementedError("Unknown method. Please choose specify one method from dp_flip and eo_flip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fair score")
    config = yaml_config_hook("./config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action='store_true',
                        help="run a couple of epochs with a small batch,"
                             "resulting accuracy is irrelevant")
    args = parser.parse_args()

    if args.debug:
        args.n_epochs = 2
        args.batch_size = 16

    args = vars(args)
    run_expt(args, dry_run=args['debug'])
