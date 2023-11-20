try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass

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
    parser = argparse.ArgumentParser(description='DeepV')
    parser.add_argument('--probType', type=str, default='acopf57',help='problem type')
    parser.add_argument('--epochs', type=int,
        help='number of neural network epochs')
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
    parser.add_argument('--max_outer_iter', type=int, help='max number of outer iterations')
    parser.add_argument('--max_inner_iter', type=int, help='max number of inner iterations')
    parser.add_argument('--alpha', type=float, help='alpha')
    parser.add_argument('--tau', type=float, help='tau')
    parser.add_argument('--rho_max', type=float, help='rho_max')

    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.deepv_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    setproctitle('DeepV-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if 'acopf' in prob_type:
        filepath = os.path.join('datasets', 'acopf', prob_type + '_dataset')
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

    save_dir = os.path.join('results', str(data), 'method', my_hash(str(sorted(list(args.items())))),
        str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
    
    # Run method
    train_net(data, args, save_dir)


def train_net(data, args, save_dir):
    solver_step = args['lr']
    nepochs = args['epochs']
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

    # create a copy of the solver_net, with the same parameters as solver_net
    solver_net_copy = NNSolver(data, args)
    solver_net_copy.to(DEVICE)
    solver_net_copy.load_state_dict(solver_net.state_dict())
    solver_net_copy.eval()

    stats = {}
    T = 15 # total outer iterations
    I = 500 # total innter iterations

    # initialize the dual variables and the step size
    # rho = 0.1
    mu = torch.ones(data.nineq, device=DEVICE)*0.1
    lam = torch.ones(data.neq, device=DEVICE)*0.1
    

    # parameters for updating dual variables
    v = 0.0
    alpha = 2
    tau = 0.8
    rho = 1.0
    rho_max = 5000

    # inner loop: train the solver_net so that the predicted demand is close to the true demand
    # outer loop: train the solver_net so that the predicted solution is feasible, by updating the dual variables
    for t in range(nepochs):
        epoch_stats = {}
        total_ineq_dist = 0.0

        solver_net.train()
        for i in range(I):
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                solver_opt.zero_grad()
                Yhat_train, _, _ = solver_net(Xtrain)
                train_loss = total_loss(data, Xtrain, Yhat_train, mu, lam, rho)
                train_loss.sum().backward()
                solver_opt.step()
                train_time = time.time() - start_time
                with torch.no_grad():
                    total_ineq_dist += data.ineq_dist(Xtrain, Yhat_train).sum(dim=0)
                    
                dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
                dict_agg(epoch_stats, 'train_time', train_time, op='sum')

            # Get valid loss
            solver_net.eval()
            for Xvalid in valid_loader:
                Xvalid = Xvalid[0].to(DEVICE)
                eval_net(data, Xvalid, solver_net, args, 'valid', epoch_stats, mu, lam, rho)

            if i%100 == 0:
                # Get test loss
                solver_net.eval()
                for Xtest in test_loader:
                    Xtest = Xtest[0].to(DEVICE)
                    eval_net(data, Xtest, solver_net, args, 'test', epoch_stats, mu, lam, rho)
            
            if i%50 == 0:
                print(
                    'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, eq mean {:.4f}, eq num viol {:.4f}, time {:.4f}'.format(
                        i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                        np.mean(epoch_stats['valid_ineq_max']),
                        np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0'])/data.nineq,
                        np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_eq_mean']),
                        np.mean(epoch_stats['valid_eq_num_viol_0'])/data.neq, np.mean(epoch_stats['valid_time'])))

        # update the dual variables
        with torch.no_grad():
            X = data.trainX.to(DEVICE)
            Y, _, _ = solver_net(X)
            eq_resid = data.eq_resid(X, Y)
            ineq_resid = data.ineq_resid(X, Y)
            ineq_dist = data.ineq_dist(X, Y)


            mu = torch.clamp(mu + rho*torch.max(ineq_dist, dim=0)[0], min=0)
            lam = lam + rho*(torch.max(torch.clamp(eq_resid, 0), dim=0)[0]- torch.max(torch.clamp(-eq_resid, 0), dim=0)[0])

            sigma = torch.max(ineq_resid, -mu/rho)
            v_new = torch.max(torch.max(torch.abs(eq_resid)), torch.max(torch.abs(sigma)))
            if v_new > tau * v:
                rho = min(alpha * rho, rho_max)


        print("outer iteration: {}, rho: {}, lam: {}".format(t, rho, lam.abs().max().item()))

        # update the parameters of the solver_net_copy
        solver_net_copy.load_state_dict(solver_net.state_dict())

    # with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
    #     pickle.dump(stats, f)
    # with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
    #     torch.save(solver_net.state_dict(), f)
    return solver_net, stats



def total_loss(data, X, Y, mu, lam, rho):
    obj_cost = data.obj_fn(Y)
    ineq_resid = data.ineq_resid(X, Y)
    eq_resid = data.eq_resid(X, Y)
    ineq_dist = data.ineq_dist(X, Y)

    return obj_cost + torch.sum(mu*ineq_resid, dim=1) + torch.sum(lam*eq_resid, dim=1) \
        + rho/2*(torch.sum(ineq_dist**2, dim=1) + torch.sum(eq_resid**2, dim=1))

def demand_loss(Pd, Qd, Pd_pred, Qd_pred):
    demand = torch.cat((Pd, Qd), dim=1)
    pred_demand = torch.cat((Pd_pred, Qd_pred), dim=1)
    return torch.sum((pred_demand - demand)**2, dim=1)


######### Models

class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        # layers = reduce(operator.add,
        #     [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ELU(), nn.Dropout(p=0.1)]
        #         for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.ELU()]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], data.output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
 
        out = nn.Hardsigmoid()(out)   # used to interpolate between max and min values
        return self._data.complete_partial(x, out)
    

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
def eval_net(data, X, solver_net, args, prefix, stats, mu, lam, rho):
    eps_converge = 1e-4
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y, _, _ = solver_net(X)
    end_time = time.time()

    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Y, mu, lam, rho).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Y).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Y), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Y) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Y)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_0'),
             torch.sum(torch.abs(data.eq_resid(X, Y)) > eps_converge, dim=1).detach().cpu().numpy())
    return stats



if __name__=='__main__':
    main()
