import torch
import numpy as np

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
from qcqp_utils import QCQPProbem
from dcopf_utils import DcopfProblem
import default_args
from model_utils import NNSolver_DC3

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='method_eq_proj')
    parser.add_argument('--probType', type=str, default='simple', help='problem type')
        # choices=['simple', 'nonconvex', 'acopf57', 'convex_qcqp', 'dcopf']
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
    parser.add_argument('--qcqpVar', type=int,
        help='number of decision vars for nonconvex problem')
    parser.add_argument('--qcqpIneq', type=int,
        help='number of inequality constraints for nonconvex problem')
    parser.add_argument('--qcqpEq', type=int,
        help='number of equality constraints for nonconvex problem')
    parser.add_argument('--qcqpEx', type=int,
        help='total number of datapoints for nonconvex problem')
    parser.add_argument('--epochs', type=int,
        help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int,
        help='training batch size')
    parser.add_argument('--lr', type=float,
        help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int,
        help='hidden layer size for neural network')
    parser.add_argument('--corrEps', type=float,
        help='correction procedure tolerance')
    parser.add_argument('--softWeight', type=float,
        help='total weight given to constraint violations in loss')
    parser.add_argument('--softWeightEqFrac', type=float,
        help='fraction of weight given to equality constraints (vs. inequality constraints) in loss')
    parser.add_argument('--saveAllStats', type=str_to_bool,
        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
        help='how frequently (in terms of number of epochs) to save stats to file')
    
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
    parser.add_argument('--corrLr', type=float,
        help='learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float,
        help='momentum for correction procedure')

    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.method_eq_proj_default_args(args['probType'])
    # defaults = default_args.method_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    setproctitle('method_eq_proj-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        torch.set_default_dtype(torch.float64)
        filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    elif prob_type == 'acopf57':
        filepath = os.path.join('datasets', 'acopf', 'acopf57_dataset')
    elif prob_type in ['convex_qcqp']:
        filepath = os.path.join('datasets', prob_type, "random_qcqp_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['qcqpVar'], args['qcqpIneq'], args['qcqpEq'], args['qcqpEx']))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = QCQPProbem(dataset)
    elif 'dcopf' in prob_type:
        filepath = os.path.join('datasets', 'dcopf', prob_type+"_data")
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = DcopfProblem(dataset)
    else:
        raise NotImplementedError
    if prob_type not in ['convex_qcqp'] and 'dcopf' not in prob_type:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)


    eval_dir = {}
    eval_dir["method_eq_proj"] = "/home/jxxiong/A-xjx/DC3/results/SimpleProblem-100-50-50-10000/method_eq_proj/3be51f559fa45158c10455c512599a7ac1cf45fd/1712043240-684846/"
    eval_dir["method"] = "/home/jxxiong/A-xjx/DC3/results/SimpleProblem-100-50-50-10000/method/b74be682712a541f619f246f722dd019a58c45dd/1711952823-7119317/"


    solver_net = NNSolver_DC3(data, args)
    # solver_net.load_state_dict(torch.load(os.path.join(eval_dir['method'], "solver_net.dict")))
    solver_net.load_state_dict(torch.load(os.path.join(eval_dir['method_eq_proj'], "solver_net.dict")))

    test_dataset = TensorDataset(data.testX)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    epoch_stats = {}
    solver_net.eval()
    for Xtest in test_loader:
        Xtest = Xtest[0]
        eval_net_cpu(data, Xtest, solver_net, args, 'test', epoch_stats)

    with open(os.path.join(eval_dir["method_eq_proj"], 'stats.dict'), 'rb') as f:
        stats = pickle.load(f)
    for key in epoch_stats.keys():
        stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
    
    print(stats)
    print(stats['cpu_test_time'])

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

def eval_net_cpu(data, X, solver_net, args, prefix, stats):
    make_prefix = lambda x: "{}_{}_{}".format('cpu', prefix, x)
    X_np = X.to('cpu')
    solver_net.to('cpu')

    total_time = 0
    for x in X_np:
        x = x.unsqueeze(0)
        start_time = time.time()
        Y = solver_net(x)
        Ycorr = data.projection(x, Y)
        end_time = time.time()
        total_time += end_time - start_time
        x.to(data.device)
        Ycorr.to(data.device)
        dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(x, Ycorr), dim=1)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(x, Ycorr), dim=1).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_max'),
                torch.max(torch.abs(data.eq_resid(x, Ycorr)), dim=1)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(x, Ycorr)), dim=1).detach().cpu().numpy())

    dict_agg(stats, make_prefix('time'), total_time / len(X_np))
    return stats


# class NNSolver(nn.Module):
#     def __init__(self, data, args):
#         super().__init__()
#         self._data = data
#         self._args = args
#         layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
#         layers = reduce(operator.add, 
#             [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
#                 for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
#         layers += [nn.Linear(layer_sizes[-1], data.ydim)]
#         for layer in layers:
#             if type(layer) == nn.Linear:
#                 nn.init.kaiming_normal_(layer.weight)

#         self.net = nn.Sequential(*layers)
        
#     def forward(self, x):
#         prob_type = self._args['probType']
#         if prob_type == 'simple':
#             return self.net(x)
#         elif prob_type == 'nonconvex':
#             return self.net(x)
#         elif prob_type == 'convex_qcqp':
#             return self.net(x)
#         elif 'acopf' in prob_type:
#             out = self.net(x)
#             data = self._data
#             out2 = nn.Sigmoid()(out[:, :-data.nbus])
#             pg = out2[:, :data.ng] * data.pmax + (1-out2[:, :data.ng]) * data.pmin
#             qg = out2[:, data.ng:2*data.ng] * data.qmax + (1-out2[:, data.ng:2*data.ng]) * data.qmin
#             vm = out2[:, 2*data.ng:] * data.vmax + (1- out2[:, 2*data.ng:]) * data.vmin
#             return torch.cat([pg, qg, vm, out[:, -data.nbus:]], dim=1)
#         elif 'dcopf' in prob_type:
#             out = self.net(x)
#             data = self._data
#             out[:, data.bounded_index] = nn.functional.sigmoid((out[:, data.bounded_index])) * data.Ub[data.bounded_index] + (1 - nn.functional.sigmoid((out[:, data.bounded_index]))) * data.Lb[data.bounded_index]
#             out[:, data.ub_only_index] = -nn.functional.relu(out[:, data.ub_only_index]) + data.Ub[data.ub_only_index] 
#             # out[:, data.lb_only_index] = nn.functional.relu(out[:, data.lb_only_index] - self.Lb[data.lb_only_index]) + self.Lb[data.lb_only_index]
#             out[:, data.lb_only_index] = nn.functional.relu(out[:, data.lb_only_index]) + data.Lb[data.lb_only_index]

#             return out
#         else:
#             raise NotImplementedError

if __name__ == "__main__":
    main()