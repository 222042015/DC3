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
    
    # Run method
    # train_net(data, args, save_dir)
    # with wandb.init(project=prob_type+'_'+args['sample'], config=args, name="multi_converge"):
    #     config = wandb.config
    #     solver_net, stats = train_net(data, args, save_dir)
    solver_net, stats = train_net(data, args, save_dir)

def train_net(data, args, save_dir):
    solver_step = args['lr']
    nepochs = 3000 #args['epochs']
    batch_size = 50
    

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_net = NNSolver(data, args)
    solver_net.to(DEVICE) 
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)
    # scheduler = optim.lr_scheduler.StepLR(solver_opt, step_size=20, gamma=0.95)
    # solver_step = 1e-8
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(solver_opt, T_max=100, eta_min=5e-6)

    solver_step = 0.0001
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(solver_opt, T_max=200, eta_min=0.00001)
    scheduler = optim.lr_scheduler.ExponentialLR(solver_opt, gamma=0.95)

    # load the weights from pure NN network
    # try:
    #     state_dict = torch.load('model_weights.pth')
    # except:
    #     raise ValueError('No model weights found')
    
    # for name, param in solver_net.named_parameters():
    #     if 'output' in name:
    #         param.data = state_dict[name][data._partial_unknown_vars]
    #     elif name in state_dict:
    #         param.data = state_dict[name]

    # solver_net.load_state_dict(torch.load("/home/jxxiong/A-xjx/DC3/results/ACOPF-57-truncated_normal-0-0.5-0.7-1000-50-50/pre_trained/solver_net.pth"))
    # # only fine-tune the last layer
    # for name, param in solver_net.named_parameters():
    #     if 'output' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    # solver_net_copy = NNSolver(data, args)
    # solver_net_copy.to(DEVICE)
    # solver_net_copy.load_state_dict(solver_net.state_dict())
    # solver_net_copy.eval()
            
    # base_solver = NNSolver_Base(data, args)
    # base_solver.to(DEVICE)
    # base_solver.load_state_dict(state_dict)
    # base_solver.eval()

    stats = {}
    factor = 10.0
    flag = True
    iters = 0
    args['factor'] = factor * torch.ones(data.nineq, device=DEVICE, requires_grad=False)
    # wandb.watch(solver_net, log="all")


    best_val = 999999

    for i in range(nepochs):
        epoch_stats = {}
        # Get train loss
        
        solver_net.train()
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
        

            Yhat_prob = solver_net(Xtrain)
            Yhat_partial_train = data.scale_partial(Yhat_prob)
            Yhat_train, converged = data.complete_partial2(Xtrain, Yhat_partial_train)
            
            # if converged.sum() > 0:
            #     Xtrain_conv = Xtrain[converged.bool(), :]
            #     Yhat_train_conv = Yhat_train[converged.bool(), :]
            #     train_loss = total_loss(data, Xtrain_conv, Yhat_train_conv, args)
            
            # weighted_loss = train_loss * adapt_weight
            # weighted_loss.sum().backward()
            # solver_opt.step()

            # dict_agg(epoch_stats, 'train_loss_obj', train_loss[0].sum().unsqueeze(0).detach().cpu().numpy())
            # dict_agg(epoch_stats, 'train_loss_ineq', train_loss[1].sum().unsqueeze(0).detach().cpu().numpy())
            
            train_loss = total_loss(data, Xtrain, Yhat_train, args)
            solver_opt.zero_grad()
            train_loss.sum().backward()
            solver_opt.step()
            # scheduler.step()
            train_time = time.time() - start_time
            if i > 50 and i % 10 == 0:
                update_t(data, Xtrain, Yhat_train, args)
            dict_agg(epoch_stats, 'train_loss', train_loss.sum().unsqueeze(0).detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')
            dict_agg(epoch_stats, 'train_nonconverged', torch.sum(converged == 0).unsqueeze(0).detach().cpu().numpy())

        if (i+1) % 100 == 0:
            print('maximum t: ', args['factor'].max().item())
            print('minimum t: ', args['factor'].min().item())
        
        if (i+1) % 50 == 0:
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
            'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, test eval {:.4f}, test ineq max {:.4f}'.format(
                i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                np.mean(epoch_stats['valid_ineq_max']),
                np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
                np.mean(epoch_stats['test_eval']), np.mean(epoch_stats['test_ineq_max'])))

        # print(
        #     'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, time {:.4f}'.format(
        #         i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
        #         np.mean(epoch_stats['valid_ineq_max']),
        #         np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
        #         np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_time'])))
        # wandb.log({'train_loss': np.mean(epoch_stats['train_loss']), 
        #            'valid_eval': np.mean(epoch_stats['valid_eval']), 
        #            'valid_ineq_max': np.mean(epoch_stats['valid_ineq_max']), 
        #            'valid_ineq_mean': np.mean(epoch_stats['valid_ineq_mean']), 
        #            'valid_eq_max': np.mean(epoch_stats['valid_eq_max']), 
        #            'train_nonconverged': np.mean(epoch_stats['train_nonconverged']),
        #            'valid_time': np.mean(epoch_stats['valid_time']),
        #            'train_time': np.mean(epoch_stats['train_time']),
        #            'valid_ineq_num_viol': np.mean(epoch_stats['valid_ineq_num_viol_0'])})

        if np.mean(epoch_stats['valid_eval']) < best_val:
            best_val = np.mean(epoch_stats['valid_eval'])
            torch.save(solver_net.state_dict(), os.path.join(save_dir, 'solver_net_val.pth'))

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
            # torch.save(solver_net.state_dict(), os.path.join(wandb.run.dir, 'solver_net.pth'))
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
                torch.save(solver_net.state_dict(), f)
            
            torch.save(solver_net.state_dict(), os.path.join(save_dir, 'solver_net_1000.pth'))
            # torch.save(solver_net.state_dict(), os.path.join(wandb.run.dir, 'solver_net_1000.pth'))
        
    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
        torch.save(solver_net.state_dict(), f)
    
    torch.save(solver_net.state_dict(), os.path.join(save_dir, 'solver_net.pth'))
    # torch.save(solver_net.state_dict(), os.path.join(wandb.run.dir, 'solver_net.pth'))
    print(save_dir)
    return solver_net, stats

def log_barrier_vectorized(z, t):
    # t_tensor = torch.ones(z.shape, device=DEVICE) * t
    t_tensor = t
    return torch.where(z <= -1 / t**2, -torch.log(-z) / t, t * z - torch.log(1 / (t_tensor**2)) / t + 1 / t)

def total_loss(data, X, Y, args):
    t = args['factor']
    obj_cost = data.obj_fn(Y).mean(dim=0) 

    ineq_resid = data.ineq_resid(X, Y)
    ineq_resid_bar = torch.zeros_like(ineq_resid)
    for i in range(ineq_resid.shape[0]):
        ineq_resid_bar[i] = log_barrier_vectorized(ineq_resid[i], t)
    ineq_resid_bar = ineq_resid_bar.sum(dim=1).sum(dim=0)
    # ineq_resid_bar = ineq_resid_bar.mean(dim=0)

    loss = torch.cat((ineq_resid_bar.unsqueeze(0), obj_cost.unsqueeze(0)))

    return loss[0]

def update_t(data, X, Y, args):
    t = args['factor']
    ineq_resid = data.ineq_dist(X, Y)
    ineq_resid_mean = ineq_resid.mean(dim=0)
    # multiply t by 1.05 if the component of ineq_resid_mean is greater than 0.1
    for i in range(ineq_resid_mean.shape[0]):
        if ineq_resid_mean[i] > 0.001:
            t[i] = min(1e4, t[i] * 1.01)
    args['factor'] = t


######### Models
# class NNSolver(nn.Module):
#     def __init__(self, data, args):
#         super().__init__()
#         self._data = data
#         self._args = args
#         layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]

#         output_dim = data.ydim - data.nknowns - data.neq

#         dic = []
#         for i in range(len(layer_sizes)-1):
#             dic.append((f'linear{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1])))
#             dic.append((f'batchnorm{i}', nn.BatchNorm1d(layer_sizes[i+1])))
#             dic.append((f'activation{i}', nn.ELU()))
#             dic.append((f'dropout{i}', nn.Dropout(p=0.1)))

#         dic.append((f'linear_output', nn.Linear(layer_sizes[-1], output_dim)))
#         self.net = nn.Sequential(OrderedDict(dic))

#         for name, param in self.net.named_parameters():
#             if "linear" in name and "weight" in name:
#                 nn.init.kaiming_normal_(param)

#     def forward(self, x):
#         prob_type = self._args['probType']
#         if prob_type == 'simple':
#             return self.net(x)
#         elif prob_type == 'nonconvex':
#             return self.net(x)
#         elif 'acopf' in prob_type:
#             out = self.net(x)
#             return nn.Sigmoid()(out)

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
            dic.append((f'activation{i}', nn.ELU()))
        
        self.net = nn.Sequential(OrderedDict(dic))

        dic_pg = []
        dic_pg.append((f'linear_pg', nn.Linear(layer_sizes[-1], 100)))
        dic_pg.append((f'activation_pg', nn.ELU()))
        dic_pg.append((f'linear_pg_output', nn.Linear(100, len(data.pg_pv_zidx))))

        self.net_pg = nn.Sequential(OrderedDict(dic_pg))
    
        dic_spv = []
        dic_spv.append((f'linear_pv', nn.Linear(layer_sizes[-1], 100)))
        dic_spv.append((f'activation_pv', nn.ELU()))
        dic_spv.append((f'linear_pv_output', nn.Linear(100, len(data.vm_spv_zidx))))
        
        self.net_spv = nn.Sequential(OrderedDict(dic_spv))

        for name, param in self.net.named_parameters():
            if "linear" in name and "weight" in name:
                nn.init.kaiming_normal_(param)
        
        for name, param in self.net_pg.named_parameters():
            if "linear" in name and "weight" in name:
                nn.init.kaiming_normal_(param)

        for name, param in self.net_spv.named_parameters():
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
            pg = self.net_pg(out)
            spv = self.net_spv(out)
            output =  torch.cat((pg, spv), dim=1)
            output = nn.Sigmoid()(output)   # used to interpolate between max and min values
            return output
        
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
    Y, converged = data.complete_partial2(X, Y_partial)
    end_time = time.time()

    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    # dict_agg(stats, make_prefix('loss'), total_loss(data, X, Y, args).detach().cpu().numpy())
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

class NNSolver_Base(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]

        dic = []
        for i in range(len(layer_sizes)-1):
            dic.append((f'linear{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1])))
            # dic.append((f'batchnorm{i}', nn.BatchNorm1d(layer_sizes[i+1])))
            dic.append((f'activation{i}', nn.ELU()))
            dic.append((f'dropout{i}', nn.Dropout(p=0.1)))

        dic.append((f'linear_output', nn.Linear(layer_sizes[-1], data.ydim)))
        # dic.append((f'activation{len(self.num_hidden_list)}', nn.Softplus()))
        self.net = nn.Sequential(OrderedDict(dic))

        for name, param in self.net.named_parameters():
            if "weight" in name and "linear" in name:
                nn.init.kaiming_normal_(param)

        
    def forward(self, x):
        prob_type = self._args['probType']
        if prob_type == 'simple':
            return self.net(x)
        elif prob_type == 'nonconvex':
            return self.net(x)
        elif 'acopf' in prob_type:
            out = self.net(x)
            data = self._data
            out2 = nn.Sigmoid()(out[:, :-data.nbus])
            pg = out2[:, :data.ng] * data.pmax + (1-out2[:, :data.ng]) * data.pmin
            qg = out2[:, data.ng:2*data.ng] * data.qmax + (1-out2[:, data.ng:2*data.ng]) * data.qmin
            vm = out2[:, 2*data.ng:] * data.vmax + (1- out2[:, 2*data.ng:]) * data.vmin
            return torch.cat([pg, qg, vm, out[:, -data.nbus:]], dim=1)
        else:
            raise NotImplementedError


if __name__=='__main__':
    main()
