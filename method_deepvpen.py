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
# DEVICE = torch.device("cpu")

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

    setproctitle('DeepVPen-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if 'acopf' in prob_type:
        filepath = os.path.join('datasets', 'acopf_v', prob_type + '_dataset')
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

    save_dir = os.path.join('results', str(data), 'method_deepvpen', my_hash(str(sorted(list(args.items())))),
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
    solver_step = 5e-4
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)


    dual_net = Dual_NN(data, args)
    dual_step = 5e-4 #1e-4(118)
    dual_net.to(DEVICE)
    dual_opt = optim.Adam(dual_net.parameters(), lr=dual_step)

    dual_net_copy = Dual_NN(data, args)
    dual_net_copy.to(DEVICE)
    dual_net_copy.load_state_dict(dual_net.state_dict())
    dual_net_copy.eval()

    stats = {}
    T = 20 # total outer iterations
    I = 150 # total innter iterations
    I_warmup = 200

    # parameters for updating dual variables
    rho = 1

    args['rho'] = rho

    rho_max = 30000
    alpha = 5 # 5-118
    v_prev = -float('inf')
    step = 0
    gamma = 0.005

    for t in range(T+1):
        
        if t == 0:
            inner_iter = I_warmup
        elif t == T:
            inner_iter = 500
        else:
            inner_iter = I 

        solver_net.train()
        dual_net.eval()
        for i in range(inner_iter):
            epoch_stats = {}
            v = -float('inf')
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                solver_opt.zero_grad()
                Yhat_train = solver_net(Xtrain)
                if t == 0:
                    lam = torch.zeros((Xtrain.shape[0], data.nineq), device=DEVICE)
                else:
                    lam = dual_net(Xtrain)
                train_loss = total_loss(data, Xtrain, Yhat_train, lam, rho)
                train_loss.sum().backward()
                solver_opt.step()
                train_time = time.time() - start_time
                with torch.no_grad():
                    ineq_resid = data.ineq_resid(Xtrain, Yhat_train)
                    sigma = torch.max(-lam/rho, ineq_resid)
                    v = max(v, torch.max(torch.norm(sigma, dim=1, p=float('inf'))).detach().cpu().numpy())
                    
                dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
                dict_agg(epoch_stats, 'train_time', train_time, op='sum')

            
            # Get valid loss
            solver_net.eval()
            dual_net.eval()
            for Xvalid in valid_loader:
                Xvalid = Xvalid[0].to(DEVICE)
                eval_net(data, Xvalid, solver_net, dual_net, 'valid', epoch_stats, rho)

            if i%1 == 0:
                # Get test loss
                solver_net.eval()
                dual_net.eval()
                for Xtest in test_loader:
                    Xtest = Xtest[0].to(DEVICE)
                    eval_net(data, Xtest, solver_net, dual_net, 'test', epoch_stats, rho)
            
            if i%10 == 0:
                print(
                    'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, eq mean {:.4f}, eq num viol {:.4f}, time {:.4f}'.format(
                        i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                        np.mean(epoch_stats['valid_ineq_max']),
                        np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0'])/data.nineq,
                        np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_eq_mean']),
                        np.mean(epoch_stats['valid_eq_num_viol_0'])/data.neq, np.mean(epoch_stats['valid_time'])))
                print('train ineq: {}'.format(torch.max(data.ineq_dist(Xtrain, Yhat_train), dim=1)[0].mean().detach().cpu().numpy()))
                # print('va_min: {}, va_max: {}'.format(va_min, va_max))

            if args['saveAllStats']:
                if t == 0 and i == 0:
                    for key in epoch_stats.keys():
                        stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
                else:
                    for key in epoch_stats.keys():
                        stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
            else:
                stats = epoch_stats

            
            if step % args['resultsSaveFreq'] == 0:
                with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                    pickle.dump(stats, f)
                with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
                    torch.save(solver_net.state_dict(), f)

            step += 1
        # dual learning
        dual_net_copy.load_state_dict(dual_net.state_dict())
        dual_net.train()
        solver_net.eval()
        for i in range(100):
            dual_epoch_stats = {}
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                Yhat_train = solver_net(Xtrain)
                dual_opt.zero_grad()
                lam_pred = dual_net(Xtrain)
                if t == 0:
                    lam_prev = torch.zeros((Xtrain.shape[0], data.nineq), device=DEVICE)
                else:
                    lam_prev = dual_net_copy(Xtrain)
                dual_total_loss = dual_loss(data, Xtrain, Yhat_train, lam_prev, lam_pred, rho)
                dual_total_loss.sum().backward()
                dual_opt.step()

                dict_agg(dual_epoch_stats, 'dual_loss', dual_total_loss.detach().cpu().numpy(), )

            if i % 10 == 0:
                print('Epoch {}: dual loss {:.4f}'.format(i, np.mean(dual_epoch_stats['dual_loss'])))

        # update rho
        with torch.no_grad():
            if v > 0.6 * v_prev:
                rho = min(alpha * rho, rho_max)
        v_prev = v
            

        I += 5
        # if t % 5 == 0:
        #     # update the learning rate of the solver
        #     solver_step = solver_step * 0.995
        #     solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)

        #     dual_step = dual_step * 0.995
        #     dual_opt = optim.Adam(dual_net.parameters(), lr=dual_step)

        # for param_group in solver_opt.param_groups:
        #     param_group['lr'] = solver_step / (1 + gamma*t)
        
        # for param_group in dual_opt.param_groups:
        #     param_group['lr'] = dual_step / (1 + gamma*t)

        print("outer iteration: {}, step: {}, rho: {}".format(t, step, rho))

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
        torch.save(solver_net.state_dict(), f)

    return solver_net, stats

def total_loss(data, X, Y, lam, rho):
    obj_cost = data.obj_fn(Y) 
    ineq_dist = data.ineq_dist(X, Y)
    ineq_resid = data.ineq_resid(X, Y)
    eq_dist = data.eq_dist(X, Y)
    return obj_cost + torch.sum(lam * ineq_resid, dim=1) + rho / 2 * (torch.sum(ineq_dist**2, dim=1) + torch.sum(eq_dist**2, dim=1))
    # return obj_cost + torch.sum(lam * ineq_resid, dim=1) + rho / 2 * (torch.sum(ineq_dist**2, dim=1))



def dual_loss(data, X, Y, lam, lam_pred, rho):
    ineq_resid = data.ineq_resid(X, Y)
    # print(torch.clamp(lam+rho*ineq_resid, min=0).max().item())
    return torch.norm(lam_pred-torch.clamp(lam+rho*ineq_resid, min=0), dim=1)


class Dual_NN(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize']]
        # layer_sizes = [data.xdim, 50, 50]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.GELU()]
                        for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], data.nineq)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return nn.ReLU()(self.net(x))

######### Models

class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.ELU(), nn.Dropout(p=0.1)]
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
def eval_net(data, X, solver_net, dual_net, prefix, stats, rho):
    eps_converge = 1e-4
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y = solver_net(X)
    end_time = time.time()

    lam = dual_net(X)

    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Y, lam, rho).detach().cpu().numpy())
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
    
    dict_agg(stats, make_prefix('raw_time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('raw_eval'), data.obj_fn(Y).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Y)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(data.ineq_dist(X, Y), dim=1).detach().cpu().numpy())
    return stats



if __name__=='__main__':
    main()
