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
    parser = argparse.ArgumentParser(description='pdl')
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
    parser.add_argument('--saveAllStats', type=str_to_bool,
        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
        help='how frequently (in terms of number of epochs) to save stats to file')
    
    parser.add_argument('--max_outer_iter', type=int)
    parser.add_argument('--max_inner_iter', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--rho_max', type=float)
    parser.add_argument('--rho', type=float)
    parser.add_argument('--v', type=float)
    
    parser.add_argument('--prefix', type=str, default='/data1/jxxiong/DC3/',
                        help='directory to the results')

    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.pdl_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    setproctitle('PDL-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    elif 'acopf' in prob_type:
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

    print("number of samples: {}".format(data.num))

    prefix = args['prefix']
    save_dir = os.path.join(prefix + 'results', str(data), 'method_pdl', my_hash(str(sorted(list(args.items())))),
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
    outer_iter = args['max_outer_iter']
    inner_iter = args['max_inner_iter']

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


    # define the primal and dual networks
    if "acopf" in args["probType"]:
        primal = Primal_NN_ACOPF(data, args)
    else:
        primal = Primal_NN(data, args)
    primal.to(DEVICE)
    
    dual = Dual_NN(data, args)
    dual.to(DEVICE)

    # make a copy of the dual network and froze the parameters
    dual_copy = Dual_NN(data, args)
    dual_copy.to(DEVICE)
    dual_copy.eval()
    dual_copy.load_state_dict(dual.state_dict())
    # freeze the parameters of the dual_copy network
    for param in dual_copy.parameters():
        param.requires_grad = False

    primal_opt = optim.Adam(primal.parameters(), lr=solver_step)
    dual_opt = optim.Adam(dual.parameters(), lr=solver_step)

    stats = {}
    # store the best validation loss
    best_primal_valid_loss = float('inf')
    best_dual_valid_loss = float('inf')

    for k in range(outer_iter):
        dual.eval()
        dual_copy.eval()
        # train the primal network
        for l in range(inner_iter): # inner for loop for l epochs
            epoch_stats = {}
            primal.eval()
            for Xvalid in valid_loader:
                Xvalid = Xvalid[0].to(DEVICE)
                eval_primal_net(data, Xvalid, primal, dual, args, 'valid', epoch_stats)
                
            primal.eval()
            for Xtest in test_loader:
                Xtest = Xtest[0].to(DEVICE)
                eval_primal_net(data, Xtest, primal, dual, args, 'test', epoch_stats)
            
            primal.train()
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                primal_opt.zero_grad()
                Yhat_train = primal(Xtrain)
                # calculate the dual variables
                duals = dual_copy(Xtrain)
                # calculate the primal loss
                primal_loss_train = primal_loss(data, Xtrain, Yhat_train, duals, args)
                primal_loss_train.sum().backward()
                primal_opt.step()
                train_time = time.time() - start_time
                dict_agg(epoch_stats, 'train_loss', primal_loss_train.detach().cpu().numpy())
                dict_agg(epoch_stats, 'train_time', train_time, op='sum')
            
            if l % 10 == 0:
                print('Training P: Outer iter {}, inner epoch {}: primal train loss {:.4f}, learning rate {:.4e}'.format(k, l, np.mean(epoch_stats['train_loss']), primal_opt.param_groups[0]['lr']))

            # saving the stats, if saveAllStats, then save all stats for each epoch, otherwise, only save the stats for the latest epoch
            if args['saveAllStats']:
                if l == 0 and k == 0:
                    for key in epoch_stats.keys():
                        stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
                else:
                    for key in epoch_stats.keys():
                        stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
            else:
                stats = epoch_stats
                
        # update the learning rate for primal network
        # validate on the validation set
        primal.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_primal_net(data, Xvalid, primal, dual, args, 'valid', epoch_stats)
        # if the average validation loss is greater than the best validation loss, thrink the learning rate by a factor of 0.99
        if np.mean(epoch_stats['valid_loss']) > best_primal_valid_loss:
            for param_group in primal_opt.param_groups:
                param_group['lr'] *= 0.99
        else:
            best_primal_valid_loss = np.mean(epoch_stats['valid_loss'])
        print('Validation: Outer iter {}, inner epoch {}: primal validation loss {:.4f}, learning rate {:.4e}'.format(k, l, np.mean(epoch_stats['valid_loss']), primal_opt.param_groups[0]['lr']))

        # train the dual network
        # dual_copy.load_state_dict(dual.state_dict())
        primal.eval()
        for l in range(inner_iter):
            dual_epoch_stats = {}
            dual.train()
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                Yhat_train = primal(Xtrain)
                dual_opt.zero_grad()
                duals = dual_copy(Xtrain)
                duals_pred = dual(Xtrain)
                dual_loss_train = dual_loss(data, Xtrain, Yhat_train, duals, duals_pred, args)
                dual_loss_train.sum().backward()
                dual_opt.step()
                train_time = time.time() - start_time
                dict_agg(dual_epoch_stats, 'train_dual_loss', dual_loss_train.detach().cpu().numpy())
                dict_agg(dual_epoch_stats, 'train_dual_time', train_time, op='sum')
            
            if l % 10 == 0:
                print('Training D: Outer iter {}, inner epoch {}: dual train loss {:.4f}, learning rate {:.4e}'.format(k, l, np.mean(dual_epoch_stats['train_dual_loss']), dual_opt.param_groups[0]['lr']))

        dual.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_dual_net(data, Xvalid, primal, dual, dual_copy, args, 'valid', dual_epoch_stats)
            
        if np.mean(dual_epoch_stats['valid_dual_loss']) > best_dual_valid_loss:
            for param_group in dual_opt.param_groups:
                param_group['lr'] *= 0.99
        else:
            best_dual_valid_loss = np.mean(dual_epoch_stats['valid_dual_loss'])
        print('Validation: Outer iter {}, inner epoch {}: dual validation loss {:.4f}, learning rate {:.4e}'.format(k, l, np.mean(dual_epoch_stats['valid_dual_loss']), dual_opt.param_groups[0]['lr']))


        # update the dual_copy network with the parameters in dual
        dual_copy.load_state_dict(dual.state_dict())

        print("Test epoch {}: loss {:.4f}, eval {:.4f}, ground_truth {:.4f}, ineq_max {:.4f}, ineq_mean {:.4f}, eq_max {:.4f}, eq_mean {:.4f}".format(\
            k, np.mean(epoch_stats['test_loss']), np.mean(epoch_stats['test_eval']), np.mean(data.obj_fn(data.testY).detach().cpu().numpy()), \
            np.mean(epoch_stats['test_ineq_max']), np.mean(epoch_stats['test_ineq_mean']), np.mean(epoch_stats['test_eq_max']), np.mean(epoch_stats['test_eq_mean'])))
        

        # update the penalty parameter rho
        update_rho(data, primal, dual, args)
        # 

        if (k % args['resultsSaveFreq'] == 0):
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, 'primal_net.dict'), 'wb') as f:
                torch.save(primal.state_dict(), f)
            with open(os.path.join(save_dir, 'dual_net.dict'), 'wb') as f:
                torch.save(dual.state_dict(), f)
            

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'primal_net.dict'), 'wb') as f:
        torch.save(primal.state_dict(), f)
    with open(os.path.join(save_dir, 'dual_net.dict'), 'wb') as f:
        torch.save(dual.state_dict(), f)
        
    # save the solution into .mat file
    with torch.no_grad():
        primal.eval()
        for Xtest in test_loader:
            Xtest = Xtest[0].to(DEVICE)
            Ytest = primal(Xtest)    
    
    with open(os.path.join(save_dir, 'sol.dict'), 'wb') as f:
        pickle.dump(Ytest.detach().cpu().numpy(), f)

    print(save_dir)
    return primal, dual, stats
        

def eval_primal_net(data, X, primal_nn, dual_nn, args, prefix, stats):
    start_time = time.time()
    Y = primal_nn(X)
    total_time = time.time() - start_time
    duals = dual_nn(X)
    make_prefix = lambda x: "{}_{}".format(prefix, x)
    dict_agg(stats, make_prefix('loss'), primal_loss(data, X, Y, duals, args).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Y).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Y), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'), torch.max(torch.abs(data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Y)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('time'), total_time, op='sum')
    return stats

def eval_dual_net(data, X, primal_nn, dual_nn, dual_nn_prev, args, prefix, stats):
    Y = primal_nn(X)
    duals = dual_nn_prev(X)
    dual_pred = dual_nn(X)
    make_prefix = lambda x: "{}_{}_{}".format(prefix,'dual', x)
    dict_agg(stats, make_prefix('loss'), dual_loss(data, X, Y, duals, dual_pred, args).detach().cpu().numpy())

    return stats
    

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


def primal_loss(data, X, Y, duals, args):
    rho = args["rho"]
    obj_cost = data.obj_fn(Y)
    ineq_dist = data.ineq_dist(X, Y)
    ineq_resid = data.ineq_resid(X, Y)
    eq_resid = data.eq_resid(X, Y)
    # calculate the sum of the square of each element of ineq_dist
    ineq_pen = torch.norm(ineq_dist, dim=1) ** 2
    eq_pen = torch.norm(eq_resid, dim=1) ** 2
    # slice the first data.eqdim elements of duals to get mu
    mu = duals[:, :data.nineq]
    lam = duals[:, data.nineq:]
    loss1 = mu.unsqueeze(1).bmm(ineq_resid.unsqueeze(2)).squeeze()
    loss2 = lam.unsqueeze(1).bmm(eq_resid.unsqueeze(2)).squeeze()
    return obj_cost + loss1 + loss2 + \
            rho/2*(ineq_pen + eq_pen)

def dual_loss(data, X, Y, duals, duals_pred, args):
    rho = args["rho"]
    ineq_resid = data.ineq_resid(X, Y)
    eq_resid = data.eq_resid(X, Y)
    mu = duals[:, :data.nineq]
    lam = duals[:, data.nineq:]
    mu_pred = duals_pred[:, :data.nineq]
    lam_pred = duals_pred[:, data.nineq:]
    # return torch.norm(mu_pred-torch.clamp(mu+rho*ineq_resid, min=0), dim=1) + \
    #         torch.norm(lam_pred-torch.clamp(lam+rho*eq_resid, min=0), dim=1)
    return torch.norm(mu_pred-torch.clamp(mu+rho*ineq_resid, min=0), dim=1) + \
            torch.norm(lam_pred-(lam+rho*eq_resid), dim=1)

def update_rho(data, primal, dual, args):
    X = data.trainX
    Y = primal(X)
    duals = dual(X)
    rho_max = args['rho_max']
    rho = args['rho']
    tau = args['tau']
    v_prev = args['v']
    alpha = args['alpha']
    eq_resid = data.eq_resid(X, Y)
    ineq_resid = data.ineq_resid(X, Y)
    lam = duals[:, data.neq:]
    # let sigma be the larger value of each element of lam and eq_norm
    sigma = torch.max(-lam/rho, ineq_resid)
    v = torch.max(torch.max(torch.norm(eq_resid, dim=1, p=float('inf')), torch.norm(sigma, dim=1, p=float('inf')))).detach().cpu().numpy()
    if v > tau * v_prev:
        args['rho'] = min(rho_max, rho*alpha)
        print("Updating rho from {:.4f} to {:.4f}".format(rho, args["rho"]))
    args['v'] = v

class Primal_NN(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                        for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], data.ydim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
                
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Dual_NN(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                        for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], data.neq + data.nineq)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
        
        layers[-1].weight.data.fill_(0)
        layers[-1].bias.data.fill_(0)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)



######### acopf network #########
class output_layers(nn.Module):
    def __init__(self, data, args, out_type="generator"):
        super().__init__()
        self._data = data
        self._args = args

        if out_type == "generator":
            self.output_size = data.ng
        elif out_type == "bus":
            self.output_size = data.nbus
        elif out_type == "branch":
            self.output_size = data.nbranch
        else:
            raise NotImplementedError

        self.linear1 = nn.Linear(self._args["hiddenSize"], self.output_size)
        self.linear2 = nn.Linear(self.output_size, self.output_size)

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.hardsigmoid(x)
        return x

class Primal_NN_ACOPF(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add, [[nn.Linear(a, b), nn.ReLU()]
                                       for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)
        # create 4 subnetworks from the output of previous layers, with activation function
        self.pg_layer = output_layers(data, args, out_type="generator")
        self.qg_layer = output_layers(data, args, out_type="generator")
        self.vm_layer = output_layers(data, args, out_type="bus")
        self.va_layer = output_layers(data, args, out_type="branch")
    
    def forward(self, x):
        x = self.net(x)
        pg = self.pg_layer(x)
        qg = self.qg_layer(x)
        vm = self.vm_layer(x)
        va = self.va_layer(x)

        # transform to real values
        pg = self._data.pmin + pg * (self._data.pmax - self._data.pmin)
        qg = self._data.qmin + qg * (self._data.qmax - self._data.qmin)
        vm = self._data.vmin + vm * (self._data.vmax - self._data.vmin)
        va = self._data.angmin + va * (self._data.angmax - self._data.angmin)

        # concate the output of the 4 subnetworks
        x = torch.cat([pg, qg, vm, va], dim=1)

        return x
    
if __name__=='__main__':
    main()