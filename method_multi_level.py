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

from utils import my_hash, str_to_bool, ACOPFProblem
import default_args

import wandb
from collections import OrderedDict

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='DC3')
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
    parser.add_argument('--useTrainCorr', type=str_to_bool, default=False,
        help='whether to use correction during training')
    parser.add_argument('--useTestCorr', type=str_to_bool, default=False,
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

    parser.add_argument('--sample', type=str, default='truncated_normal', #truncated_normal
        help='how to sample data for acopf problems')

    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.method_default_args(args['probType'])
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
    elif 'acopf' in prob_type:
        if args['sample'] == 'uniform':
            filepath = os.path.join('datasets', 'acopf', prob_type+'_dataset')
        elif args['sample'] == 'truncated_normal':
            filepath = os.path.join('datasets', 'acopf_T', prob_type+'_dataset')
        else:
            raise NotImplementedError
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
    
    name = "DC3"
    if args["useTrainCorr"] == False and args["useTestCorr"] == False:
        name = "DC3_nocorr"

    save_dir = os.path.join('results', str(data), 'method', my_hash(str(sorted(list(args.items())))),
        str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
    
    # Run method
    # with wandb.init(project=prob_type+'_'+args['sample'], config=args, name=name):
    #     args = wandb.config
    #     solver_net, stats = train_net(data, args, save_dir)
    solver_net, stats = train_net(data, args, save_dir)


def train_net(data, args, save_dir):
    solver_step = 1e-3 #args['lr']
    nepochs = 1500 #args['epochs']
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
    # solver_net.load_state_dict(torch.load("/home/jxxiong/A-xjx/DC3/results/ACOPF-57-truncated_normal-0-0.5-0.7-1000-50-50/method/0243a6c6b36f7abea981e28be63cc2aab97516fd/1710252359-2775788/model_weights.pth"))

    # # only train the last layer with 'output' in the name
    # for name, param in solver_net.named_parameters():
    #     if "output" not in name:
    #         param.requires_grad = False
    args['n_ineq'] = 0.1
    stats = {}
    weight_decay = 0.0001
    for i in range(nepochs):
        epoch_stats = {}

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

        # Get train loss
        solver_net.train()
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
            solver_opt.zero_grad()
            Yhat_train = solver_net(Xtrain)
            Ynew_train = grad_steps(data, Xtrain, Yhat_train, args)
            train_loss = total_loss(data, Xtrain, Ynew_train, args)
            train_loss.sum().backward()
            solver_opt.step()
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')

        print(
            'Epoch {}: train loss {:.4f}, eval {:.4f}, dist {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, steps {}, time {:.4f}, test eval {:.4f}, test ineq max {:.4f},  ineq max less {:.4f}'.format(
                i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                np.mean(epoch_stats['valid_dist']), np.mean(epoch_stats['valid_ineq_max']),
                np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
                np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_steps']), np.mean(epoch_stats['valid_time']),
                np.mean(epoch_stats['test_eval']), np.mean(epoch_stats['test_ineq_max']),
                np.mean(epoch_stats['test_ineq_max_less'])))
        # print(
        #     'Epoch {}: train loss {:.4f}, eval {:.4f}, dist {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, steps {}, time {:.4f}, test eval {:.4f}, test ineq max {:.4f}'.format(
        #         i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
        #         np.mean(epoch_stats['valid_dist']), np.mean(epoch_stats['valid_ineq_max']),
        #         np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
        #         np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_steps']), np.mean(epoch_stats['valid_time']),
        #         np.mean(epoch_stats['test_eval']), np.mean(epoch_stats['test_ineq_max'])))

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
        
        if i % 100 == 0 and i >= 100:
            args['n_ineq'] += 0.1
            solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step*0.1, weight_decay=weight_decay*0.1)
            print('n_ineq:', min(data.ng, int(args['n_ineq'] * data.ng)))
            

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
        torch.save(solver_net.state_dict(), f)
    
    torch.save(solver_net.state_dict(), os.path.join(save_dir, 'model_weights.pth'))
    print(save_dir)
    # wandb.finish()
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
def eval_net(data, X, solver_net, args, prefix, stats):
    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y = solver_net(X)
    base_end_time = time.time()

    Ycorr, steps = grad_steps_all(data, X, Y, args)
    end_time = time.time()

    Ynew = grad_steps(data, X, Y, args)
    raw_end_time = time.time()

    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('steps'), np.array([steps]))
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Ynew, args).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max_less'), torch.max(data.ineq_dist2(X, Ycorr, args['n_ineq']), dim=1)[0].detach().cpu().numpy())
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

# def total_loss(data, X, Y, args):
#     obj_cost = data.obj_fn(Y)
#     ineq_dist = data.ineq_dist(X, Y)
#     ineq_cost = torch.norm(ineq_dist, dim=1)
#     eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
#     return obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + \
#             args['softWeight'] * args['softWeightEqFrac'] * eq_cost

def total_loss(data, X, Y, args):
    obj_cost = data.obj_fn(Y)
    ineq_dist = data.ineq_dist2(X, Y, args['n_ineq'])
    ineq_cost = torch.norm(ineq_dist, dim=1)
    return obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost

def grad_steps(data, X, Y, args):
    take_grad_steps = args['useTrainCorr']
    if take_grad_steps:
        lr = args['corrLr']
        num_steps = args['corrTrainSteps']
        momentum = args['corrMomentum']
        partial_var = args['useCompl']
        partial_corr = True if args['corrMode'] == 'partial' else False
        if partial_corr and not partial_var:
            assert False, "Partial correction not available without completion."
        Y_new = Y
        old_Y_step = 0
        for i in range(num_steps):
            if partial_corr:
                Y_step = data.ineq_partial_grad(X, Y_new)
            else:
                ineq_step = data.ineq_grad(X, Y_new)
                eq_step = data.eq_grad(X, Y_new)
                Y_step = (1 - args['softWeightEqFrac']) * ineq_step + args['softWeightEqFrac'] * eq_step
            
            # if torch.max(torch.abs(Y_step)) > 1e16:
            #     break

            new_Y_step = lr * Y_step + momentum * old_Y_step
            Y_new = Y_new - new_Y_step

            old_Y_step = new_Y_step

        return Y_new
    else:
        return Y

# Used only at test time, so let PyTorch avoid building the computational graph
def grad_steps_all(data, X, Y, args):
    take_grad_steps = args['useTestCorr']
    if take_grad_steps:
        lr = args['corrLr']
        eps_converge = args['corrEps']
        max_steps = args['corrTestMaxSteps']
        momentum = args['corrMomentum']
        partial_var = args['useCompl']
        partial_corr = True if args['corrMode'] == 'partial' else False
        if partial_corr and not partial_var:
            assert False, "Partial correction not available without completion."
        Y_new = Y
        i = 0
        old_Y_step = 0
        old_ineq_step = 0
        old_eq_step = 0
        with torch.no_grad():
            while (i == 0 or torch.max(torch.abs(data.eq_resid(X, Y_new))) > eps_converge or
                           torch.max(data.ineq_dist(X, Y_new)) > eps_converge) and i < max_steps:
                if partial_corr:
                    Y_step = data.ineq_partial_grad(X, Y_new)
                else:
                    ineq_step = data.ineq_grad(X, Y_new)
                    eq_step = data.eq_grad(X, Y_new)
                    Y_step = (1 - args['softWeightEqFrac']) * ineq_step + args['softWeightEqFrac'] * eq_step
                
                # if torch.max(torch.abs(Y_step)) > 1e16:
                #     break

                new_Y_step = lr * Y_step + momentum * old_Y_step
                Y_new = Y_new - new_Y_step

                old_Y_step = new_Y_step
                i += 1

        return Y_new, i
    else:
        return Y, 0


######### Models
class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]

        output_dim = data.ydim - data.nknowns - data.neq

        dic = []
        # for i in range(len(layer_sizes)-1):
        #     dic.append((f'linear{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1])))
        #     dic.append((f'batchnorm{i}', nn.BatchNorm1d(layer_sizes[i+1])))
        #     dic.append((f'activation{i}', nn.ELU()))
        #     dic.append((f'dropout{i}', nn.Dropout(p=0.1)))
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
            output = nn.Sigmoid()(out)   # used to interpolate between max and min values
            return self._data.complete_partial(x, output)

# class NNSolver(nn.Module):
#     def __init__(self, data, args):
#         super().__init__()
#         self._data = data
#         self._args = args
#         layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
#         layers = reduce(operator.add,
#             [[nn.Linear(a,b), 
#             #   nn.BatchNorm1d(b), 
#               nn.ELU(), 
#               nn.Dropout(p=0.1)]
#                 for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        
#         output_dim = data.ydim - data.nknowns

#         if self._args['useCompl']:
#             layers += [nn.Linear(layer_sizes[-1], output_dim - data.neq)]
#         else:
#             layers += [nn.Linear(layer_sizes[-1], output_dim)]

#         for layer in layers:
#             if type(layer) == nn.Linear:
#                 nn.init.kaiming_normal_(layer.weight)

#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.net(x)
 
#         if self._args['useCompl']:
#             if 'acopf' in self._args['probType']:
#                 out = nn.Sigmoid()(out)   # used to interpolate between max and min values
#             return self._data.complete_partial(x, out)
#         else:
#             return self._data.process_output(x, out)

if __name__=='__main__':
    main()
