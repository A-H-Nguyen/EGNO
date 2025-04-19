import argparse
from argparse import Namespace
import torch
import torch.utils.data
from torchinfo import summary
from motion.dataset import MotionDynamicsDataset as MotionDataset
from model.egno import EGNO
import model.basic as basic
import model.layer_no as layer_no
import os
import sys
from torch import nn, optim
import json

import random
import numpy as np

import time

from utils import EarlyStopping

import hooks

parser = argparse.ArgumentParser(description='EGNO')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='exp_results', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--model', type=str, default='egno', metavar='N')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--delta_frame', type=int, default=30,
                    help='Number of frames delta.')
parser.add_argument('--data_dir', type=str, default='',
                    help='Data directory.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--config_by_file", default=None, nargs="?", const='', type=str, )

parser.add_argument('--lambda_link', type=float, default=1,
                    help='The weight of the linkage loss.')
parser.add_argument('--n_cluster', type=int, default=3,
                    help='The number of clusters.')
parser.add_argument('--flat', action='store_true', default=False,
                    help='flat MLP')
parser.add_argument('--interaction_layer', type=int, default=3,
                    help='The number of interaction layers per block.')
parser.add_argument('--pooling_layer', type=int, default=3,
                    help='The number of pooling layers in EGPN.')
parser.add_argument('--decoder_layer', type=int, default=1,
                    help='The number of decoder layers.')

parser.add_argument('--case', type=str, default='walk',
                    help='The case, walk or run.')

parser.add_argument('--num_timesteps', type=int, default=1,
                    help='The number of time steps.')
parser.add_argument('--time_emb_dim', type=int, default=32,
                    help='The dimension of time embedding.')
parser.add_argument('--num_modes', type=int, default=2,
                    help='The number of modes.')


time_exp_dic = {'time': 0, 'counter': 0}


args = parser.parse_args()
if args.config_by_file is not None:
    if len(args.config_by_file) == 0:
        job_param_path = './configs/config_mocap_no.json'
    else:
        job_param_path = args.config_by_file
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        # Only update existing keys
        args = vars(args)
        args.update((k, v) for k, v in hyper_params.items() if k in args)
        args = Namespace(**args)

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss(reduction='none')

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

# torch.autograd.set_detect_anomaly(True)

def layerwise_sparsity(model):
    print("Per-layer Model Sparsity:")
    print("=" * 55)
    
    for name, param in model.named_parameters():
        total = param.numel()
        zeros = torch.sum(param < 1e-8).item()
        sparsity = (zeros / total) * 100 if total > 0 else 0
        
        if (sparsity > 40.0):
            print(f"{name:<40} num elements: {total:>11.2f} sparsity: {sparsity:>11.2f}%")

    print("=" * 55)

def compute_module_sparsity(model, threshold=1e-8, report_threshold=40.0):
    for name, module in model.named_modules():
        total_elements = 0
        total_zeros = 0

        for param in module.parameters(recurse=True):  # only direct params
            if param is not None:
                total_elements += param.numel()
                total_zeros += (param.abs() < threshold).sum().item()

        if total_elements > 0:
            sparsity = (total_zeros / total_elements) * 100
            if sparsity > report_threshold:
                print(f"{name:<40} num elements: {total_elements:>11,d} sparsity: {sparsity:>6.2f}%")

def compute_sparsity(model):
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()  # Total number of elements
        zero_params += torch.sum(param < 1e-8).item()  # Count of zero elements
        sparsity = zero_params / total_params if total_params > 0 else 0
    print("=" * 55)
    print(f"Total Model Sparsity: {sparsity:.2%}")
    print("=" * 55)

def main():
    # fix seed
    seed = args.seed
    random.seed(seed) np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_train = MotionDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame, case=args.case, num_timesteps=args.num_timesteps)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=0)

    dataset_val = MotionDataset(partition='val', max_samples=600, data_dir=args.data_dir,
                                delta_frame=args.delta_frame, case=args.case, num_timesteps=args.num_timesteps)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=0)

    dataset_test = MotionDataset(partition='test', max_samples=600, data_dir=args.data_dir,
                                 delta_frame=args.delta_frame, case=args.case, num_timesteps=args.num_timesteps)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=0)

    if args.model == 'egno':
        model = EGNO(n_layers=args.n_layers, in_node_nf=2, in_edge_nf=2, hidden_nf=args.nf, device=device, with_v=True,
                     flat=args.flat, activation=nn.SiLU(), use_time_conv=True, num_modes=args.num_modes,
                     num_timesteps=args.num_timesteps, time_emb_dim=args.time_emb_dim)
    else:
        raise NotImplementedError('Unknown model:', args.model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model_save_path = os.path.join(args.outf, args.exp_name, 'saved_model.pth')
    # print(f'Model saved to {model_save_path}')
    # early_stopping = EarlyStopping(patience=50, verbose=True, path=model_save_path)

    # sparsity_hooks = []
    # timing_hooks = []
    forward_hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (basic.EGNN_Layer, layer_no.TimeConv, layer_no.TimeConv_x)):
            # forward_hooks.append(module.register_forward_hook(hooks.make_running_sparsity_hook()))
            forward_hooks.append(module.register_forward_hook(hooks.make_timing_hook()))
            # hooks.append(module.register_forward_hook(timing_hook))
            # hooks.append(module.register_forward_hook(sparsity_hook))


    results = {'eval epoch': [], 'val loss': [], 'test loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    best_lp_loss = 1e8
    
    start_time = time.perf_counter()

    hooks.start_timers(model)

    for epoch in range(0, args.epochs):
        train_loss, lp_loss = train(model, optimizer, epoch, loader_train)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss, _ = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss, _ = train(model, optimizer, epoch, loader_test, backprop=False)

            results['eval epoch'].append(epoch)
            results['val loss'].append(val_loss)
            results['test loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
                best_lp_loss = lp_loss
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))
            # early_stopping(val_loss, model)
            # if early_stopping.early_stop:
            #     print("Early Stopping.")
            #     break

        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/loss.json", "w") as outfile:
            outfile.write(json_object)

            
    
    hooks.report_forward_times(model)
    # hooks.report_average_sparsities(model)

    # for name, module in model.named_modules():
    #     if hasattr(module, '_forward_time'):
    #         total_params = 0
    #         zero_params = 0
            
    #         for param in module.parameters(recurse=True):
    #             total_params += param.numel()  # Total number of elements
    #             zero_params += torch.sum(param < 1e-8).item()  # Count of zero elements
    #             sparsity = zero_params / total_params if total_params > 0 else 0
    #         print("=" * 55)
    #         print(f"{name} - Sparsity: {sparsity:.2%}, runtime: {module._forward_time}")

    # print("=" * 55)
    layerwise_sparsity(model)
    print(f"Total runtime: {time.perf_counter() - start_time} seconds")

    # print(model)
    # summary(model)
    # # compute_sparsity(model)
    # layerwise_sparsity(model)


    return best_train_loss, best_val_loss, best_test_loss, best_epoch, best_lp_loss


def train(model, optimizer, epoch, loader, backprop=True):
    
    start_time = time.perf_counter()

    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'lp_loss': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        # data = [d.view(-1, d.size(2)) for d in data]  # construct mini-batch graphs
        for i in [-1, -2]:
            d = data[i].view(batch_size * n_nodes, args.num_timesteps, 3)
            data[i] = d.transpose(0, 1).contiguous().view(-1, 3)

        loc, vel, edges, edge_attr, local_edges, local_edge_fea, Z, loc_end, vel_end = data
        # convert into graph minibatch
        loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, loc.size(2))  # [BN, 3]
        loc = loc.view(-1, loc.size(2))
        vel = vel.view(-1, vel.size(2))
        offset = (torch.arange(batch_size) * n_nodes).unsqueeze(-1).unsqueeze(-1).to(edges.device)
        edges = torch.cat(list(edges + offset), dim=-1)  # [2, BM]
        edge_attr = torch.cat(list(edge_attr), dim=0)  # [BM, ]
        local_edge_index = torch.cat(list(local_edges + offset), dim=-1)  # [2, BM]
        local_edge_fea = torch.cat(list(local_edge_fea), dim=0)  # [BM, ]
        Z = Z.view(-1, Z.size(2))

        optimizer.zero_grad()

        if args.model == 'egno':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            # loc_pred = model(nodes, loc, edges, vel, edge_attr)
            loc_pred, vel_pred, _ = model(loc, nodes, edges, edge_attr, v=vel, loc_mean=loc_mean)
        else:
            raise Exception("Wrong model")

        # loss = loss_mse(loc_pred, loc_end)
        losses = loss_mse(loc_pred, loc_end).view(args.num_timesteps, batch_size * n_nodes, 3)
        losses = torch.mean(losses, dim=(1, 2))
        loss = torch.mean(losses)

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += losses[-1].item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f avg lploss: %.5f'
          % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], res['lp_loss'] / res['counter']))

    # print(f"\nepoch {epoch} took {time.perf_counter() - start_time} seconds\n")

    return res['loss'] / res['counter'], res['lp_loss'] / res['counter']


if __name__ == "__main__":
    best_train_loss, best_val_loss, best_test_loss, best_epoch, best_lp_loss = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_lp = %.6f" % best_lp_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)
    print("best_train = %.6f, best_lp = %.6f, best_val = %.6f, best_test = %.6f, best_epoch = %d"
          % (best_train_loss, best_lp_loss, best_val_loss, best_test_loss, best_epoch))

