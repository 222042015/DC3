import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

from collections import OrderedDict
import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse

from utils import my_hash, str_to_bool, ACOPFProblem
import default_args
import random

import wandb
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='DeepV_Multi')
    parser.add_argument('--probType', type=str, default='acopf57'
                        , help='problem type')
    parser.add_argument('--simpleVar', type=int, 
        help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int,
        help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int,
        help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int,
        help='total number of datapoints for simple problem')
    parser.add_argument('--nonconvexVar', type=int,
        help='number of decision vars for nonconvex problem')
    parser.add_argument('--nonconvexIneq', type=int,
        help='number of inequality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEq', type=int,
        help='number of equality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEx', type=int,
        help='total number of datapoints for nonconvex problem')
    parser.add_argument('--epochs', type=int,
        help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int,
        help='training batch size')
    parser.add_argument('--lr', type=float,
        help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int,
        help='hidden layer size for neural network')
    parser.add_argument('--softWeight', type=float,
        help='total weight given to constraint violations in loss')
    parser.add_argument('--softWeightEqFrac', type=float,
        help='fraction of weight given to equality constraints (vs. inequality constraints) in loss')
    parser.add_argument('--useCompl', type=str_to_bool,
        help='whether to use completion')
    parser.add_argument('--useTrainCorr', type=str_to_bool,
        help='whether to use correction during training')
    parser.add_argument('--useTestCorr', type=str_to_bool,
        help='whether to use correction during testing')
    parser.add_argument('--corrMode', choices=['partial', 'full'],
        help='employ DC3 correction (partial) or naive correction (full)')
    parser.add_argument('--corrTrainSteps', type=int,
        help='number of correction steps during training')
    parser.add_argument('--corrTestMaxSteps', type=int,
        help='max number of correction steps during testing')
    parser.add_argument('--corrEps', type=float,
        help='correction procedure tolerance')
    parser.add_argument('--corrLr', type=float,
        help='learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float,
        help='momentum for correction procedure')
    parser.add_argument('--saveAllStats', type=str_to_bool,
        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
        help='how frequently (in terms of number of epochs) to save stats to file')

    parser.add_argument('--sample', type=str, default='truncated_normal',
        help='how to sample data for acopf problems')
    # parser.add_argument('--sample', type=str, default='uniform',
        # help='how to sample data for acopf problems')

    args = parser.parse_args()
    args = vars(args) # change to dictionary
    # defaults = default_args.deepv_default_args(args['probType'])
    defaults = default_args.method_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    setproctitle('DeepV_Multi-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    elif 'acopf' in prob_type:
        if args['sample'] == 'uniform':
            filepath = os.path.join('datasets', 'acopf', prob_type+'_dataset')
        elif args['sample'] == 'truncated_normal':
            filepath = os.path.join('datasets', 'acopf_T', prob_type+'_dataset')
    else:
        raise NotImplementedError

    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    data = ACOPFProblem(dataset, train_num=1000, valid_num=50, test_num=50) #, valid_frac=0.05, test_frac=0.05)
    data._device = DEVICE
    print(DEVICE)
    for attr in dir(data):
        var = getattr(data, attr)
        if torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass

    save_dir = os.path.join('results', str(data), 'method_deepv_multi', my_hash(str(sorted(list(args.items())))),
        str(time.time()).replace('.', '-'))
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)

    args['save_dir'] = save_dir
    
    solver_net = train_net(data, args, save_dir)

def train_net(data, args, save_dir):
    solver_step = 5e-4
    nepochs = 1000 #args['epochs']
    batch_size = 100
    

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_net = NNSolver(data, args)
    solver_net.to(DEVICE) 
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(solver_opt, 0.95)

    for i in range(nepochs):
        epoch_stats = {}
        # Get train loss
        
        solver_net.train()
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
        
            Yp_train, Yq_train, Yv_train = solver_net(Xtrain)

            train_loss = total_loss(data, Xtrain, Yp_train, Yq_train, Yv_train, args)
            solver_opt.zero_grad()
            train_loss.sum().backward()
            solver_opt.step()

            train_time = time.time() - start_time

            dict_agg(epoch_stats, 'train_loss', train_loss.sum().unsqueeze(0).detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')

        if i % 50 == 0:
            scheduler.step()

        # Get valid loss
        solver_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            Yp_valid, Yq_valid, Yv_valid = solver_net(Xvalid)
            valid_loss = total_loss(data, Xvalid, Yp_valid, Yq_valid, Yv_valid, args)
            dict_agg(epoch_stats, 'valid_loss', valid_loss.sum().unsqueeze(0).detach().cpu().numpy())
            dict_agg(epoch_stats, 'valid_pg_dist', torch.max(torch.clamp(data.ineq_resid_pg(Xvalid, Yp_valid), 0), dim=1)[0].detach().cpu().numpy())
            dict_agg(epoch_stats, 'valid_qg_dist', torch.max(torch.clamp(data.ineq_resid_qg(Xvalid, Yq_valid), 0), dim=1)[0].detach().cpu().numpy())
            dict_agg(epoch_stats, 'valid_v_dist', torch.max(torch.clamp(data.ineq_resid_v(Xvalid, Yv_valid), 0), dim=1)[0].detach().cpu().numpy())

        print(
            'Epoch {}: train loss {:.4f}, eval {:.4f}, valid_dist_pg {:.4f}, valid_dist_qg {:.4f}, valid_dist_v {:.4f}'.format(
                i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_loss']),
                np.mean(epoch_stats['valid_pg_dist']), np.mean(epoch_stats['valid_qg_dist']), np.mean(epoch_stats['valid_v_dist'])))

    torch.save(solver_net.base.state_dict(), os.path.join(save_dir, 'model_weights_base.pth'))
    torch.save(solver_net.head_q.state_dict(), os.path.join(save_dir, 'model_weights_head_q.pth'))
    with open(os.path.join(save_dir, 'solver_net_base.dict'), 'wb') as f:
        torch.save(solver_net.state_dict(), f)

    return solver_net
        
# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value



class NNSolver_Base(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]

        dic = []
        for i in range(len(layer_sizes) - 1):
            dic.append((f'linear{i}', nn.Linear(layer_sizes[i], layer_sizes[i + 1])))
            dic.append((f'batchnorm{i}', nn.BatchNorm1d(layer_sizes[i + 1])))
            dic.append((f'activation{i}', nn.ELU()))

        self.net = nn.Sequential(OrderedDict(dic))
        for name, param in self.net.named_parameters():
            if "linear" in name and "weight" in name:
                nn.init.kaiming_normal_(param)
    
    def forward(self, x):
        return self.net(x)

class NNsolver_Head(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        output_dim = data.ydim - data.nknowns - data.neq

        dic = []
        dic.append((f'linear_head', nn.Linear(self._args['hiddenSize'], 100)))
        dic.append((f'activation_head', nn.ELU()))

        dic.append((f'linear_head_output', nn.Linear(100, output_dim)))
        dic.append((f'activation_head_output', nn.Sigmoid()))

        self.net_head = nn.Sequential(OrderedDict(dic))
        for name, param in self.net_head.named_parameters():
            if "linear" in name and "weight" in name:
                nn.init.kaiming_normal_(param)
    
    def forward(self, x_hidden):
        out = self.net_head(x_hidden)
        return out

class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        
        self.base = NNSolver_Base(data, args)
        self.head_p = NNsolver_Head(data, args)
        self.head_q = NNsolver_Head(data, args)
        self.head_v = NNsolver_Head(data, args)
    
    def forward(self, x):
        x_hidden = self.base(x)
        p = self.head_p(x_hidden)
        y_p = self._data.complete_partial(x, p)
        q = self.head_q(x_hidden)
        y_q = self._data.complete_partial(x, q)
        v = self.head_v(x_hidden)
        y_v = self._data.complete_partial(x, v)

        return y_p, y_q, y_v

def total_loss(data, X, Yp, Yq, Yv, args):
    loss_p = torch.clamp(data.ineq_resid_pg(X, Yp), 0)
    loss_q = torch.clamp(data.ineq_resid_qg(X, Yq), 0)
    loss_v = torch.clamp(data.ineq_resid_v(X, Yv), 0)

    # return loss_p.mean() + loss_q.mean() + loss_v.mean()
    return torch.norm(loss_p, dim=1) + torch.norm(loss_q, dim=1) + torch.norm(loss_v, dim=1)

if __name__=='__main__':
    main()
