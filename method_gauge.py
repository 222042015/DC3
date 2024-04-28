# try:
#     import waitGPU
#     waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
# except ImportError:
#     pass

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse

from utils import my_hash, str_to_bool
import default_args

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='Gauge')
    parser.add_argument('--probType', type=str, default='simple',help='problem type')
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
    parser.add_argument('--batchSize', type=int,
        help='training batch size')
    parser.add_argument('--lr', type=float,
        help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int,
        help='hidden layer size for neural network')
    parser.add_argument('--saveAllStats', type=str_to_bool,
        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
        help='how frequently (in terms of number of epochs) to save stats to file')
    parser.add_argument('--useCompl', type=str_to_bool,
        help='whether to use completion')
    parser.add_argument('--corrEps', type=float,
        help='correction procedure tolerance')
    
    parser.add_argument('--inner_warmstart', type=int,
        help='number of epochs for warmstart')
    parser.add_argument('--inner_iter', type=int,
        help='number of epochs in the inner iterations')
    parser.add_argument('--outer_iter', type=int,
        help='number of outer iterations')
    parser.add_argument('--beta', type=float,
        help='increase the number of inner interations after each outer iteration')
    parser.add_argument('--rho', type=float,
        help='initial rho')
    parser.add_argument('--lambda', type=float,
        help='initial lambda')
    parser.add_argument('--gamma', type=float,
        help='Decrements of step size')
    

    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.deeplde_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    setproctitle('Gauge-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        filepath = os.path.join('datasets_gauge', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets_gauge', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    else:
        raise NotImplementedError
    # read the data and transfer to GPU
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE

    save_dir = os.path.join('results', str(data), 'method_gauge', my_hash(str(sorted(list(args.items())))),
        str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
    
    # Run method
    train_net(data, args, save_dir)


def train_net(data, args, save_dir):
    # solver_step = args['lr']
    # batch_size = args['batchSize']
    solver_step = 1e-3
    batch_size = 32
    
    train_dataset = TensorDataset(data.trainX, data.trainIP)
    valid_dataset = TensorDataset(data.validX, data.validIP)
    test_dataset = TensorDataset(data.testX, data.testIP)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_net = NNSolver(data, args)
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)
    scheduler = optim.lr_scheduler.StepLR(solver_opt, step_size=1000, gamma=0.95)

    stats = {}
    for i in range(10000):
        epoch_stats = {}
        # Get valid loss
        solver_net.eval()
        for Xvalid, IPvalid in valid_loader:
            Xvalid = Xvalid.to(DEVICE)
            IPvalid = IPvalid.to(DEVICE)
            eval_net(data, Xvalid, IPvalid, solver_net, args, 'valid', epoch_stats)
        # Get test loss
        solver_net.eval()
        for Xtest, IPtest in test_loader:
            Xtest = Xtest.to(DEVICE)
            IPtest = IPtest.to(DEVICE)
            eval_net(data, Xtest, IPtest, solver_net, args, 'test', epoch_stats)
        # Get train loss
        solver_net.train()
        for Xtrain, IPtrain in train_loader:
            Xtrain = Xtrain.to(DEVICE)
            IPtrain = IPtrain.to(DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            v_train = solver_net(Xtrain, IPtrain)
            Ypartial_train = data.gauge_map(v_train, IPtrain, Xtrain)
            Yhat_train = data.complete_partial(Xtrain, Ypartial_train)
            train_loss = total_loss(data, Xtrain, Yhat_train)
            train_loss.sum().backward()
            solver_opt.step()
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')
        
        if i % 100 == 0:
            print(
            'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, time {:.4f}'.format(
                i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                np.mean(epoch_stats['valid_ineq_max']),
                np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
                np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_time'])))
        if args['saveAllStats']:
            if i == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            stats = epoch_stats
            
            
        if (i % args['resultsSaveFreq'] == 0):
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
                torch.save(solver_net.state_dict(), f)
        scheduler.step()


    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
        torch.save(solver_net.state_dict(), f)
    return solver_net, stats

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
def eval_net(data, X, u0, solver_net, args, prefix, stats):
    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    # Y = solver_net(X)
    v = solver_net(X, u0)
    Ypartial = data.gauge_map(v, u0, X)
    Y = data.complete_partial(X, Ypartial)
    end_time = time.time()
    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Y).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Y).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Y), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Y) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Y) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Y) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Y)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_0'),
             torch.sum(torch.abs(data.eq_resid(X, Y)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_1'),
             torch.sum(torch.abs(data.eq_resid(X, Y)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_2'),
             torch.sum(torch.abs(data.eq_resid(X, Y)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    
    dict_agg(stats, make_prefix('raw_time'), end_time - start_time, op='sum')
    # dict_agg(stats, make_prefix('steps'), np.array([steps]))
    dict_agg(stats, make_prefix('raw_eval'), data.obj_fn(Y).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(data.ineq_dist(X, Y), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Y)), dim=1).detach().cpu().numpy())
    return stats


def total_loss(data, X, Y):
    obj_cost = data.obj_fn(Y)
    return obj_cost


######### Models

class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        output_dim = data.ydim - data.nknowns - data.neq
        layer_sizes = [data.xdim + output_dim, 100] #self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.ReLU()] # , nn.Dropout(p=0.1)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])

        layers += [nn.Linear(layer_sizes[-1], output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x, u0):
        out = self.net(torch.cat([x, u0], dim=1))
        return nn.Tanh()(out)
 
if __name__=='__main__':
    main()
