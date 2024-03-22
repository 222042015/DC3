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
    solver_step = 5e-4 #args['lr']
    nepochs = 3000 #args['epochs']
    batch_size = 50#args['batchSize']

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_base = NNSolver_Base(data, args)
    solver_base.to(DEVICE)
    solver_base.load_state_dict(torch.load("/home/jxxiong/A-xjx/DC3/results/ACOPF-57-truncated_normal-0-0.5-0.7-1000-50-50/method_deepv_multi/ffebf8203c4ffc2eb09d511e2a8ed8d818574a55/1710904993-5504515/model_weights_base.pth"))

    for param in solver_base.parameters():
        param.requires_grad = False
    
    solver_head = NNSolver_Head(data, args)
    solver_head.to(DEVICE)
    solver_opt = optim.Adam(solver_head.parameters(), lr=solver_step, weight_decay=1e-4)
    solver_head.load_state_dict(torch.load("/home/jxxiong/A-xjx/DC3/results/ACOPF-57-truncated_normal-0-0.5-0.7-1000-50-50/method_deepv_multi/ffebf8203c4ffc2eb09d511e2a8ed8d818574a55/1710904993-5504515/model_weights_head_q.pth"))
    scheduler = optim.lr_scheduler.ExponentialLR(solver_opt, 0.95)

    stats = {}
    for i in range(nepochs):
        epoch_stats = {}

        # Get valid loss
        solver_head.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_net(data, Xvalid, solver_head, solver_base, args, 'valid', epoch_stats)

        # Get test loss
        solver_head.eval()
        for Xtest in test_loader:
            Xtest = Xtest[0].to(DEVICE)
            eval_net(data, Xtest, solver_head, solver_base, args, 'test', epoch_stats)

        # Get train loss
        solver_head.train()
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            Yhidden = solver_base(Xtrain)
            Yprob = solver_head(Yhidden)
            Yhat_train = data.complete_partial(Xtrain, Yprob)
            train_loss = total_loss(data, Xtrain, Yhat_train, args)
            train_loss.sum().backward()
            solver_opt.step()
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')


        print(
            'Epoch {}: train loss {:.4f}, eval {:.4f}, dist {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, time {:.4f}, test eval {:.4f}, test ineq max {:.4f}'.format(
                i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                np.mean(epoch_stats['valid_dist']), np.mean(epoch_stats['valid_ineq_max']),
                np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
                np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_time']),
                np.mean(epoch_stats['test_eval']), np.mean(epoch_stats['test_ineq_max'])))

        if i % 10 == 0:
            scheduler.step()


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

# Modifies stats in place
def eval_net(data, X, solver_head, solver_base, args, prefix, stats):
    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Xhidden = solver_base(X)
    Yprob = solver_head(Xhidden)
    Y = data.complete_partial(X, Yprob)
    base_end_time = time.time()


    # Ycorr, steps = grad_steps_all(data, X, Y, args)
    end_time = time.time()

    # Ynew = grad_steps(data, X, Y, args)
    raw_end_time = time.time()

    Ycorr = Y
    Ynew = Y

    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    # dict_agg(stats, make_prefix('steps'), np.array([steps]))
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Ynew, args).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    # dict_agg(stats, make_prefix('ineq_max'), torch.max(torch.clamp(data.ineq_resid_pg(X, Ycorr), 0), dim=1)[0].detach().cpu().numpy())
    # dict_agg(stats, make_prefix('ineq_max_less'), torch.max(data.ineq_dist2(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Ycorr) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Ycorr) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_0'),
             torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_1'),
             torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_2'),
             torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_time'), (raw_end_time-end_time) + (base_end_time-start_time), op='sum')
    dict_agg(stats, make_prefix('raw_eval'), data.obj_fn(Ynew).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(data.ineq_dist(X, Ynew), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(data.ineq_dist(X, Ynew), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Ynew) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Ynew) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Ynew) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Ynew)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_mean'),
             torch.mean(torch.abs(data.eq_resid(X, Ynew)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
             torch.sum(torch.abs(data.eq_resid(X, Ynew)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
             torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
             torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    return stats




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

class NNSolver_Head(nn.Module):
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
        self.head_p = NNSolver_Head(data, args)
        self.head_q = NNSolver_Head(data, args)
        self.head_v = NNSolver_Head(data, args)
    
    def forward(self, x):
        x_hidden = self.base(x)
        p = self.head_p(x_hidden)
        y_p = self._data.complete_partial(x, p)
        q = self.head_q(x_hidden)
        y_q = self._data.complete_partial(x, q)
        v = self.head_v(x_hidden)
        y_v = self._data.complete_partial(x, v)

        return y_p, y_q, y_v

def total_loss(data, X, Y, args):
    obj_cost = data.obj_fn(Y)
    ineq_dist = data.ineq_dist(X, Y)
    ineq_cost = torch.norm(ineq_dist, dim=1)
    eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
    return ineq_cost

if __name__=='__main__':
    main()