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
import random

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='DeepV_Multi')
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

    save_dir = os.path.join('results', str(data), 'method_deepv_multi', my_hash(str(sorted(list(args.items())))),
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
    solver_opt = optim.Adam(solver_net.parameters(), lr=5e-4)

    classifier = Classifier(data, args)
    classifier.to(DEVICE)
    classifier_opt = optim.Adam(classifier.parameters(), lr=5e-4)

    classifier_copy = Classifier(data, args)
    classifier_copy.to(DEVICE)
    classifier_copy.load_state_dict(classifier.state_dict())
    for param in classifier_copy.parameters():
        param.requires_grad = False

    stats = {}
    T = 1000
    I = 1
    I_warmup = 1

    # inner loop: train the solver_net so that the predicted demand is close to the true demand
    # outer loop: train the solver_net so that the predicted solution is feasible, by updating the dual variables
    step = 0
    buffer = BalancedBuffer(30000)
    w = 100000
    train_classfier = True
    for t in range(T+1):
        
        if t == 0:
            inner_iter = I_warmup
        else:
            inner_iter = I 
        
        if t == 0:
            solver_net.train()
            classifier.eval()
            for i in range(50):
                epoch_stats = {}
                for Xtrain in train_loader:
                    Xtrain = Xtrain[0].to(DEVICE)
                    start_time = time.time()
                    solver_opt.zero_grad()
                    Yhat_prob = solver_net(Xtrain)
                    Yhat_partial_train = data.scale_partial(Yhat_prob)
                    Yhat_train, converged = data.complete_partial(Xtrain, Yhat_partial_train)
                    if converged.sum() > 0:
                        Xtrain_conv = Xtrain[converged.bool(), :]
                        Yhat_train_conv = Yhat_train[converged.bool(), :]
                        train_loss = total_loss(data, Xtrain_conv, Yhat_train_conv, t+1)
                        train_loss.mean().backward()
                        solver_opt.step()
                        train_time = time.time() - start_time
                    buffer.add(Yhat_prob, converged)

                    dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
                    dict_agg(epoch_stats, 'train_time', train_time, op='sum')

                print('Epoch {}: train loss {:.4f}, time {:.4f}'.format(i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['train_time'])))

            solver_net.eval()
            classifier.train()
            classifier_copy.eval()
            for i in range(10000):
                epoch_stats = {}
                if len(buffer.buffer_0) == 0 or len(buffer.buffer_1) == 0:
                    break
                # sample the data from the buffer
                Xtrain, Ytrain = buffer.sample(200)
                classifier_opt.zero_grad()
                pred = classifier(Xtrain).squeeze(-1)
                loss = classifier_loss(pred, Ytrain)
                loss.mean().backward()
                classifier_opt.step()
                dict_agg(epoch_stats, 'train_classifier_loss', loss.squeeze().detach().cpu().numpy())

                if i % 100 == 0:
                    print('Epoch {}: train classifier loss {:.4f}'.format(i, np.mean(epoch_stats['train_classifier_loss'])))
            
            classifier_copy.load_state_dict(classifier.state_dict())



        for i in range(5):
            epoch_stats = {}
            solver_net.train()
            classifier_copy.eval()
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                solver_opt.zero_grad()
                Yhat_prob = solver_net(Xtrain)
                Yhat_partial_train = data.scale_partial(Yhat_prob)
                Yhat_train, converged = data.complete_partial(Xtrain, Yhat_partial_train)
                solver_loss = total_loss(data, Xtrain, Yhat_train, (t+1)*5)

                # feasible = classifier(Yhat_partial_train)
                feasible = classifier_copy(Yhat_prob).squeeze(-1)
                label = torch.ones(feasible.shape, device=DEVICE, dtype=torch.long)
                class_loss = classifier_loss(feasible, label)

                train_loss = solver_loss + class_loss * w
                # train_loss = class_loss
                train_loss.mean().backward()
                # nn.utils.clip_grad_value_(solver_net.parameters(), clip_value)

                solver_opt.step()
                train_time = time.time() - start_time
                buffer.add(Yhat_prob, converged)
                    
                dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
                dict_agg(epoch_stats, 'train_time', train_time, op='sum')

            # Get valid loss
            print("evaluating")
            solver_net.eval()
            for Xvalid in valid_loader:
                Xvalid = Xvalid[0].to(DEVICE)
                eval_net(data, Xvalid, solver_net, args, 'valid', epoch_stats)
            
            if i%1 == 0:
                print(
                    'Epoch {}: train loss {:.4f}, eval {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, eq mean {:.4f}, eq num viol {:.4f}, time {:.4f}'.format(
                        i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                        np.mean(epoch_stats['valid_ineq_max']),
                        np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0'])/data.nineq,
                        np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_eq_mean']),
                        np.mean(epoch_stats['valid_eq_num_viol_0'])/data.neq, np.mean(epoch_stats['valid_time'])))


            if i == 4 and train_classfier:
                solver_net.eval()
                classifier.train()
                for s in range(5000):
                    if len(buffer.buffer_0) == 0 or len(buffer.buffer_1) == 0:
                        break
                    # sample the data from the buffer
                    Xtrain, Ytrain = buffer.sample(200)
                    classifier_opt.zero_grad()
                    pred = classifier(Xtrain).squeeze(-1)
                    loss = classifier_loss(pred, Ytrain)
                    loss.mean().backward()
                    classifier_opt.step()
                    dict_agg(epoch_stats, 'train_classifier_loss', loss.squeeze().detach().cpu().numpy())

                    if s % 1000 == 0:
                        print('Epoch {}: train classifier loss {:.4f}'.format(s, np.mean(epoch_stats['train_classifier_loss'])))
                if np.mean(epoch_stats['train_classifier_loss']) < 5e-3:
                    train_classfier = False
                    print("stop training classifier")
        
        if t % 5 == 0:
            classifier_copy.load_state_dict(classifier.state_dict())
        
        if t % 30 == 0:
            # decay the learning rate of the classifier
            for param_group in classifier_opt.param_groups:
                param_group['lr'] *= 0.999
            
            for param_group in solver_opt.param_groups:
                param_group['lr'] *= 0.999

            w *= 0.8
            
        if args['saveAllStats']:
            if t == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            stats = epoch_stats
        
        if t % args['resultsSaveFreq'] == 0:
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
                torch.save(solver_net.state_dict(), f)
        step += 1

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
        torch.save(solver_net.state_dict(), f)

    return solver_net, stats

def log_barrier(z, t):
    if z <= - 1 / t**2:
        return - torch.log(-z) / t
    else:
        return t * z + -np.log(1 / (t**2)) / t + 1 / t

def log_barrier_vectorized(z, t):
    t_tensor = torch.ones(z.shape, device=DEVICE) * t
    return torch.where(z <= -1 / t**2, -torch.log(-z) / t, t * z + -torch.log(1 / (t_tensor**2)) / t + 1 / t)

def total_loss(data, X, Y, t):
    factor = np.floor(t / 50) + 1
    ineq_resid = data.ineq_resid(X, Y)
    obj_cost = data.obj_fn(Y) * factor
    ineq_resid_bar = torch.zeros_like(ineq_resid)
    t = min(t*10, 50000)
    for i in range(ineq_resid.shape[0]):
        ineq_resid_bar[i] = log_barrier_vectorized(ineq_resid[i], t)

    return ineq_resid_bar.sum(dim=1) + obj_cost


def classifier_loss(pred, label):
    return -label * torch.log(pred) - (1 - label) * torch.log(1 - pred)


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
        layers += [nn.Linear(layer_sizes[-1], data.output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
 
        out = nn.Sigmoid()(out)   # used to interpolate between max and min values
        # return self._data.complete_partial(x, out)
        return out
    

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
    # dict_agg(stats, make_prefix('loss'), total_loss(data, X, Y, lam, rho).detach().cpu().numpy())
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


class Classifier(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        output_dim = data.ydim - data.nknowns - data.neq

        layers = [nn.Linear(output_dim, 200), nn.LeakyReLU(), nn.Linear(200, 1), nn.Sigmoid()]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, y_partial):
        out = self.net(y_partial)
        return out
    
class BalancedBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_0 = []
        self.buffer_1 = []

    def add(self, samples, labels):
        for i in range(samples.size(0)):
            sample = samples[i].detach()
            label = labels[i].item()
            if label == 0:
                if len(self.buffer_0) >= self.capacity:
                    self.buffer_0.pop(0)
                self.buffer_0.append(sample)
            else:
                if len(self.buffer_1) >= self.capacity:
                    self.buffer_1.pop(0)
                self.buffer_1.append(sample)

    def sample(self, batch_size):
        batch_size_per_class = batch_size // 2
        samples_0 = torch.stack(random.sample(self.buffer_0, min(len(self.buffer_0), batch_size_per_class)))
        samples_1 = torch.stack(random.sample(self.buffer_1, min(len(self.buffer_1), batch_size_per_class)))
        # generate teh labels
        labels_0 = torch.zeros(samples_0.shape[0], device=DEVICE)
        labels_1 = torch.ones(samples_1.shape[0], device=DEVICE)
        sample = torch.cat((samples_0, samples_1), 0)
        labels = torch.cat((labels_0, labels_1), 0)
        shuffle_idx = torch.randperm(sample.shape[0])
        return sample[shuffle_idx], labels[shuffle_idx]



if __name__=='__main__':
    main()
