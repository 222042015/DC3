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

from utils import my_hash, str_to_bool, ACOPFProblem2
import default_args
import random

import wandb

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='DeepV_Multi')
    parser.add_argument('--probType', type=str, default='acopf39'
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

    # parser.add_argument('--sample', type=str, default='truncated_normal',
    #     help='how to sample data for acopf problems')
    parser.add_argument('--sample', type=str, default='uniform',
        help='how to sample data for acopf problems')

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
    data = ACOPFProblem2(dataset, train_num=1000, valid_num=50, test_num=50) #, valid_frac=0.05, test_frac=0.05)
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
    
    # Run method
    # train_net(data, args, save_dir)
    with wandb.init(project=prob_type+'_'+args['sample'], config=args, name="multi_converge"):
        config = wandb.config
        solver_net, stats = train_net(data, args, save_dir)

def train_net(data, args, save_dir):
    solver_step = args['lr']
    nepochs = 1000 #args['epochs']
    batch_size = 200
    solver_step = 0.0005

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_net = NNSolver(data, args)
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)
    scheduler = optim.lr_scheduler.StepLR(solver_opt, step_size=20, gamma=0.95)
    # load the weights from pure NN network
    # try:
    #     # state_dict = torch.load('model_weights.pth')
    #     state_dict = torch.load("/home/jxxiong/A-xjx/DC3/wandb/run-20240305_154658-3qjdpkjm/files/model_weights.pth")
    # except:
    #     raise ValueError('No model weights found')
    
    # for name, param in solver_net.named_parameters():
    #     if 'output' in name:
    #         param.data = state_dict[name][data._partial_unknown_vars]
    #     elif name in state_dict:
    #         param.data = state_dict[name]

    # solver_net.load_state_dict(torch.load("/home/jxxiong/A-xjx/DC3/wandb/run-20240301_233540-liyqkk29/files/solver_net.pth"))

    stats = {}
    factor = 1
    flag = True
    iters = 0
    args['factor'] = factor
    wandb.watch(solver_net, log="all")


    for i in range(nepochs):
        epoch_stats = {}
        # Get train loss
        
        solver_net.train()
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
        

            Yhat_prob = solver_net(Xtrain)
            Yhat_partial_train = data.scale_partial(Yhat_prob)
            Yhat_train, converged = data.complete_partial(Xtrain, Yhat_partial_train)
            
            if converged.sum() > 0:
                Xtrain_conv = Xtrain[converged.bool(), :]
                Yhat_train_conv = Yhat_train[converged.bool(), :]
                train_loss = total_loss(data, Xtrain_conv, Yhat_train_conv, args)

            solver_net.zero_grad()
            train_loss.sum().backward()
            solver_opt.step()
            
            # if flag: # reinitialize weights after updating the factor
            #     weights = torch.ones_like(train_loss)
            #     weights = torch.nn.Parameter(weights)
            #     T = weights.sum().detach()
            #     optimizer_gradnorm = torch.optim.Adam([weights], lr=0.01)
            #     l0 = train_loss.detach()
            #     flag = False
        
            
            # weighted_loss = (weights * train_loss).sum()
            # solver_opt.zero_grad()
            # weighted_loss.backward(retain_graph=True)

            # gw = []
            # for ii in range(len(train_loss)):
            #     dl = torch.autograd.grad(weights[ii] * train_loss[ii], solver_net.parameters(), retain_graph=True, create_graph=True)[0]
            #     gw.append(torch.norm(dl))
            
            # gw = torch.stack(gw)
            # loss_ratio = train_loss.detach() / l0
            # rt = loss_ratio / loss_ratio.mean()
            # gw_avg = gw.mean().detach()
            # constant = (gw_avg * rt ** 0.01).detach()
            # gradnorm_loss = torch.abs(gw - constant).sum()
            # optimizer_gradnorm.zero_grad()
            # gradnorm_loss.backward()
            # optimizer_gradnorm.step()
            # solver_opt.step()
            # optimizer_gradnorm.step()

            # weights = (T * weights / weights.sum()).detach()
            # weights = torch.nn.Parameter(weights)
            # optimizer_gradnorm = torch.optim.Adam([weights], lr=0.01)
            # iters += 1
            
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.sum().unsqueeze(0).detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')
            dict_agg(epoch_stats, 'train_nonconverged', torch.sum(converged == 0).unsqueeze(0).detach().cpu().numpy())

        scheduler.step()
        # Get valid loss
        solver_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_net(data, Xvalid, solver_net, args, 'valid', epoch_stats)

        # Get test loss
        solver_net.eval()
        for Xtest in test_loader:
            Xtest = Xtest[0].to(DEVICE)
            eval_net(data, Xtest, solver_net, args, 'test', epoch_stats)

        print(
            'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, time {:.4f}'.format(
                i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                np.mean(epoch_stats['valid_ineq_max']),
                np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
                np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_time'])))
        wandb.log({'train_loss': np.mean(epoch_stats['train_loss']), 
                   'valid_eval': np.mean(epoch_stats['valid_eval']), 
                   'valid_ineq_max': np.mean(epoch_stats['valid_ineq_max']), 
                   'valid_ineq_mean': np.mean(epoch_stats['valid_ineq_mean']), 
                   'valid_eq_max': np.mean(epoch_stats['valid_eq_max']), 
                   'train_nonconverged': np.mean(epoch_stats['train_nonconverged']),
                   'valid_time': np.mean(epoch_stats['valid_time']),
                   'train_time': np.mean(epoch_stats['train_time']),
                   'valid_ineq_num_viol': np.mean(epoch_stats['valid_ineq_num_viol_0'])})

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
            torch.save(solver_net.state_dict(), os.path.join(wandb.run.dir, 'solver_net.pth'))
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
                torch.save(solver_net.state_dict(), f)
            
            torch.save(solver_net.state_dict(), os.path.join(save_dir, 'solver_net_1000.pth'))
            torch.save(solver_net.state_dict(), os.path.join(wandb.run.dir, 'solver_net_1000.pth'))
        
        if (i+1) % 20 == 0:
            args['factor'] = min(1e7, args['factor'] * 1.2)
            flag = True

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
        torch.save(solver_net.state_dict(), f)
    
    torch.save(solver_net.state_dict(), os.path.join(save_dir, 'solver_net.pth'))
    torch.save(solver_net.state_dict(), os.path.join(wandb.run.dir, 'solver_net.pth'))
    return solver_net, stats

def log_barrier_vectorized(z, t):
    t_tensor = torch.ones(z.shape, device=DEVICE) * t
    return torch.where(z <= -1 / t**2, -torch.log(-z) / t, t * z - torch.log(1 / (t_tensor**2)) / t + 1 / t)

def total_loss(data, X, Y, args):
    t = args['factor']
    obj_cost = data.obj_fn(Y).mean(dim=0) * 10

    ineq_resid = data.ineq_resid(X, Y)
    ineq_resid_bar = torch.zeros_like(ineq_resid)
    for i in range(ineq_resid.shape[0]):
        ineq_resid_bar[i] = log_barrier_vectorized(ineq_resid[i], t)
    ineq_resid_bar = ineq_resid_bar.sum(dim=1).mean(dim=0)

    loss = torch.cat((ineq_resid_bar.unsqueeze(0), obj_cost.unsqueeze(0)))

    return loss


######### Models
class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]

        output_dim = data.ydim - data.nknowns - data.neq

        dic = []
        for i in range(len(layer_sizes)-1):
            dic.append((f'linear{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1])))
            dic.append((f'batchnorm{i}', nn.BatchNorm1d(layer_sizes[i+1])))
            dic.append((f'activation{i}', nn.ReLU()))
            dic.append((f'dropout{i}', nn.Dropout(p=0.2)))

        dic.append((f'linear_output', nn.Linear(layer_sizes[-1], output_dim)))
        self.net = nn.Sequential(OrderedDict(dic))

        for name, param in self.net.named_parameters():
            if "linear" in name and "weight" in name:
                nn.init.kaiming_normal_(param)

    def forward(self, x):
        prob_type = self._args['probType']
        if prob_type == 'simple':
            return self.net(x)
        elif prob_type == 'nonconvex':
            return self.net(x)
        elif 'acopf' in prob_type:
            out = self.net(x)
            return nn.Sigmoid()(out)

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
def eval_net(data, X, solver_net, args, prefix, stats):
    eps_converge = 1e-4
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y_prob = solver_net(X)
    Y_partial = data.scale_partial(Y_prob)
    Y, converged = data.complete_partial(X, Y_partial)
    end_time = time.time()

    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Y, args).detach().cpu().numpy())
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
