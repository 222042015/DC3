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

from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='DeepLDE')
    parser.add_argument('--probType', type=str, default='acopf57',help='problem type')
        # choices=['simple', 'nonconvex', 'acopf57', 'acopf118', 'acopf300', 'acopf1354'], help='problem type')
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
    # defaults = default_args.method_default_args(args['probType'])
    defaults = default_args.deeplde_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    setproctitle('DC3-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    # elif prob_type == 'acopf57':
    elif prob_type[:5] == 'acopf':
        filepath = os.path.join('datasets', 'acopf', prob_type + '_dataset')
        # filepath = os.path.join('datasets', 'acopf', 'acopf57_dataset')
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

    save_dir = os.path.join('results', str(data), 'method_deeplde', my_hash(str(sorted(list(args.items())))),
        str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
    
    # Run method
    train_net(data, args, save_dir)


def train_net(data, args, save_dir):
    solver_step = args['lr']
    batch_size = args['batchSize']

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_net = NNSolver(data, args)
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)

    stats = {}
    # T = 42 # total outer iterations
    # I = 25 # total innter iterations
    # I_warmup = 100
    # beta = 5 # increase the number of inner interations after each outer iteration
    # gamma = 0.01 # decrease the step size for the dual update every outer iteration

    # # initialize the dual variables and the step size
    # rho = 0.5
    # lam = torch.ones(data.nineq, device=DEVICE) * 0.1
    T = args['outer_iter']
    I = args['inner_iter']
    I_warmup = args['inner_warmstart']
    beta = args['beta']
    gamma = args['gamma']
    rho = args['rho']
    lam = torch.ones(data.nineq, device=DEVICE) * args['lambda']

    writer = SummaryWriter('runs/{}'.format(save_dir))

    step = 0
    for t in range(T+1):
        if t == 0:
            inner_iter = I_warmup
        else:
            inner_iter = I

        for i in range(inner_iter):
            epoch_stats = {}
            # Get valid loss
            solver_net.eval()
            for Xvalid in valid_loader:
                Xvalid = Xvalid[0].to(DEVICE)
                eval_net(data, Xvalid, solver_net, args, 'valid', epoch_stats, lam)

            # Get test loss
            solver_net.eval()
            for Xtest in test_loader:
                Xtest = Xtest[0].to(DEVICE)
                eval_net(data, Xtest, solver_net, args, 'test', epoch_stats, lam)

            # Get train loss
            solver_net.train()
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                solver_opt.zero_grad()
                Yhat_train = solver_net(Xtrain)
                train_loss = total_loss(data, Xtrain, Yhat_train, lam)
                train_loss.sum().backward()
                solver_opt.step()
                train_time = time.time() - start_time
                dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
                dict_agg(epoch_stats, 'train_time', train_time, op='sum')
            

            print(
                'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, time {:.4f}'.format(
                    step, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                    np.mean(epoch_stats['valid_ineq_max']),
                    np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
                    np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_time'])))

            # write to tensorboard
            writer.add_scalar('train_loss', np.mean(epoch_stats['train_loss']), step)
            writer.add_scalar('valid_eval', np.mean(epoch_stats['valid_eval']), step)
            writer.add_scalar('valid_ineq_max', np.mean(epoch_stats['valid_ineq_max']), step)
            writer.add_scalar('valid_ineq_mean', np.mean(epoch_stats['valid_ineq_mean']), step)
            writer.add_scalar('valid_eq_max', np.mean(epoch_stats['valid_eq_max']), step)

            if args['saveAllStats']:
                if t == 0 and i == 0:
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
            step += 1
            # total_ineq_dist.append(epoch_ineq_dist)
            # append epoch_ineq_dist to total_ineq_dist

        with torch.no_grad():
            epoch_ineq_dist = torch.zeros(data.nineq, device=DEVICE)
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                Yhat_train = solver_net(Xtrain)
                epoch_ineq_dist += data.ineq_dist(Xtrain, Yhat_train).sum(dim=0)
            if t == 0:
                # lam = torch.ones(data.nineq, device=DEVICE) * 0.1
                lam = torch.ones(data.nineq, device=DEVICE) * args['lambda']
            else:
                lam = lam + rho * epoch_ineq_dist
            I = I + beta
            rho = rho / (1 + gamma * t)

        print("outer iteration: {}, step: {}, rho: {}, lam: {}".format(t, step, rho, lam.abs().max().item()))

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
def eval_net(data, X, solver_net, args, prefix, stats, lam):
    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y = solver_net(X)
    end_time = time.time()
    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Y, lam).detach().cpu().numpy())
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


def total_loss(data, X, Y, lam):
    obj_cost = data.obj_fn(Y)
    ineq_dist = data.ineq_dist(X, Y)
    return obj_cost + torch.sum(lam * ineq_dist, dim=1)


######### Models

class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ELU(), nn.Dropout(p=0.1)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        
        output_dim = data.ydim - data.nknowns

        if self._args['useCompl']:
            layers += [nn.Linear(layer_sizes[-1], output_dim - data.neq)]
        else:
            layers += [nn.Linear(layer_sizes[-1], output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
 
        if self._args['useCompl']:
            if 'acopf' in self._args['probType']:
                out = nn.Sigmoid()(out)   # used to interpolate between max and min values
            return self._data.complete_partial(x, out)[0]
        else:
            return self._data.process_output(x, out)

if __name__=='__main__':
    main()
