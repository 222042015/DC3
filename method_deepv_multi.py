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

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='DeepV_Multi')
    # parser = argparse.ArgumentParser(description='DC3')
    parser.add_argument('--probType', type=str, default='acopf300'
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
        filepath = os.path.join('datasets', 'acopf', prob_type+'_dataset')
    else:
        raise NotImplementedError

    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    data = ACOPFProblem2(dataset, train_num=100, valid_num=10, test_num=10) #, valid_frac=0.05, test_frac=0.05)
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
    
    # Run method
    train_net(data, args, save_dir)

def train_net(data, args, save_dir):
    solver_step = args['lr']
    nepochs = 2000 #args['epochs']
    batch_size = 100
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

    classifier = Classifier(data, args)
    classifier.to(DEVICE)
    classifier_opt = optim.Adam(classifier.parameters(), lr=5e-4)

    buffer = BalancedBuffer(5000)

    try:
        state_dict = torch.load('model_weights.pth')
    except:
        raise ValueError('No model weights found')
    
    for name, param in solver_net.named_parameters():
        if 'output' in name:
            param.data = state_dict[name][data._partial_unknown_vars]
        elif name in state_dict:
            param.data = state_dict[name]

    stats = {}
    iters = 0
    factor = 1
    args['factor'] = factor
    flag = True


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
        classifier.eval()
        for Xtrain in train_loader:
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
            
            Yhat_prob = solver_net(Xtrain)
            Yhat_partial_train = data.scale_partial(Yhat_prob)
            Yhat_train, converged = data.complete_partial(Xtrain, Yhat_partial_train)
            buffer.add(Yhat_prob, converged)
            train_loss = total_loss(data, Xtrain, Yhat_train, args)

            # if i <= 40 and converged.sum() > 0:
            #     Xtrain_conv = Xtrain[converged.bool(), :]
            #     Yhat_train_conv = Yhat_train[converged.bool(), :]
            #     train_loss = total_loss(data, Xtrain_conv, Yhat_train_conv, args)
            # else:
            #     train_loss = total_loss(data, Xtrain, Yhat_train, args)

            # # add classifer loss to the loss to drive the model to predict feasible solutions
            # if i >= 20 and len(buffer.buffer_0) > 0 and len(buffer.buffer_1) > 0:
            #     feasible = classifier(Yhat_prob).squeeze(-1)
            #     label = torch.ones(feasible.shape, device=DEVICE, dtype=torch.long)
            #     class_loss = classifier_loss(feasible, label).mean() * 100000
            #     # append the class_loss to the train_loss
            #     train_loss = torch.cat((train_loss, class_loss.unsqueeze(0)))

            # if i >= 20 and len(buffer.buffer_0) > 0 and len(buffer.buffer_1) > 0:
            if 1:
                # sample positive samples from buffer
                samples, labels = buffer.sample(batch_size)
                # replace 0 in labels with -1
                labels[labels == 0] = -1
                # compute the similarity loss between prediction and the samples
                sim_loss = nn.CosineEmbeddingLoss()(Yhat_prob[:len(samples)], samples, labels) * 1e6
                train_loss = torch.cat((train_loss, sim_loss.unsqueeze(0)))


            if flag: # reinitialize weights after updating the factor
                weights = torch.ones_like(train_loss)
                weights = torch.nn.Parameter(weights)
                T = weights.sum().detach()
                optimizer_gradnorm = torch.optim.Adam([weights], lr=0.01)
                l0 = train_loss.detach()
                flag = False
            
            weighted_loss = (weights * train_loss).sum()
            solver_opt.zero_grad()
            weighted_loss.backward(retain_graph=True)

            gw = []
            for ii in range(len(train_loss)):
                dl = torch.autograd.grad(weights[ii] * train_loss[ii], solver_net.parameters(), retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            
            gw = torch.stack(gw)
            loss_ratio = train_loss.detach() / l0
            rt = loss_ratio / loss_ratio.mean()
            gw_avg = gw.mean().detach()
            constant = (gw_avg * rt ** 0.01).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            optimizer_gradnorm.zero_grad()
            gradnorm_loss.backward()
            optimizer_gradnorm.step()
            solver_opt.step()
            optimizer_gradnorm.step()

            weights = (T * weights / weights.sum()).detach()
            weights = torch.nn.Parameter(weights)
            optimizer_gradnorm = torch.optim.Adam([weights], lr=0.01)
            iters += 1

            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.sum().unsqueeze(0).detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')

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
        
        if (i+1) % 20 == 0:
            args['factor'] = args['factor'] * 1.2
            flag = True

        # if i <= 20 or i % 2 == 0:
        #     # train the classifier
        #     classifier.train()
        #     solver_net.eval()
        #     class_loss = []
        #     for s in range(1000):
        #         if len(buffer.buffer_0) == 0 or len(buffer.buffer_1) == 0:
        #             break
        #         # sample the data from the buffer
        #         Xtrain, Ytrain = buffer.sample(200)
        #         classifier_opt.zero_grad()
        #         pred = classifier(Xtrain).squeeze(-1)
        #         loss = classifier_loss(pred, Ytrain)
        #         loss.mean().backward()
        #         classifier_opt.step()
        #         class_loss.append(loss.mean().detach().cpu().numpy())
        #         if (s+1) % 100 == 0:
        #             print('Epoch {}: train classifier loss {:.4f}'.format(s, np.mean(class_loss)))


    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
        torch.save(solver_net.state_dict(), f)
    return solver_net, stats

def log_barrier_vectorized(z, t):
    t_tensor = torch.ones(z.shape, device=DEVICE) * t
    return torch.where(z <= -1 / t**2, -torch.log(-z) / t, t * z - torch.log(1 / (t_tensor**2)) / t + 1 / t)

def total_loss(data, X, Y, args):
    t = args['factor']
    # eq_cost = torch.norm(data.eq_resid(X, Y), dim=1).mean(dim=0)
    obj_cost = data.obj_fn(Y).mean(dim=0)

    ineq_resid = data.ineq_resid(X, Y)
    ineq_resid_bar = torch.zeros_like(ineq_resid)
    for i in range(ineq_resid.shape[0]):
        ineq_resid_bar[i] = log_barrier_vectorized(ineq_resid[i], t)
    ineq_resid_bar = ineq_resid_bar.sum(dim=1).mean(dim=0)

    loss = torch.cat((ineq_resid_bar.unsqueeze(0), obj_cost.unsqueeze(0))) #, eq_cost.unsqueeze(0)))
    return loss

def eq_loss(data, X, Y):
    eq_resid = data.eq_resid(X, Y)
    return torch.norm(eq_resid, dim=1).mean(dim=0)

def classifier_loss(pred, label):
    pred = torch.clamp(pred, 1e-7, 1 - 1e-7)

    return -label * torch.log(pred) - (1 - label) * torch.log(1 - pred)

def ineq_loss(data, X, Y, args):
    t = args['factor']
    ineq_resid = data.ineq_resid(X, Y)
    ineq_resid_bar = torch.zeros_like(ineq_resid)
    for i in range(ineq_resid.shape[0]):
        ineq_resid_bar[i] = log_barrier_vectorized(ineq_resid[i], t)
    ineq_resid_bar = ineq_resid_bar.sum(dim=1).mean(dim=0)

    return ineq_resid_bar


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
        #     data = self._data
        #     out2 = nn.Sigmoid()(out[:, :-data.nbus])
        #     pg = out2[:, :data.ng] * data.pmax + (1-out2[:, :data.ng]) * data.pmin
        #     qg = out2[:, data.ng:2*data.ng] * data.qmax + (1-out2[:, data.ng:2*data.ng]) * data.qmin
        #     vm = out2[:, 2*data.ng:] * data.vmax + (1- out2[:, 2*data.ng:]) * data.vmin
        #     return torch.cat([pg, qg, vm, out[:, -data.nbus:]], dim=1)
        # else:
        #     raise NotImplementedError

# class NNSolver(nn.Module):
#     def __init__(self, data, args):
#         super().__init__()
#         self._data = data
#         self._args = args
#         layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
#         layers = reduce(operator.add,
#             [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
#                 for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        
#         output_dim = data.ydim - data.nknowns - data.neq

#         layers += [nn.Linear(layer_sizes[-1], output_dim)]

#         for layer in layers:
#             if type(layer) == nn.Linear:
#                 nn.init.kaiming_normal_(layer.weight)

#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.net(x)
 
#         if self._args['useCompl']:
#             if 'acopf' in self._args['probType']:
#                 out = nn.Sigmoid()(out)   # used to interpolate between max and min values

#         return out

# class NNSolver(nn.Module):
#     def __init__(self, data, args):
#         super().__init__()
#         self._data = data
#         self._args = args
#         layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
#         layers = reduce(operator.add,
#             [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.1)]
#                 for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
#         layers += [nn.Linear(layer_sizes[-1], data.output_dim)]

#         for layer in layers:
#             if type(layer) == nn.Linear:
#                 nn.init.kaiming_normal_(layer.weight)

#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.net(x)
 
#         out = nn.Sigmoid()(out)   # used to interpolate between max and min values
#         # return self._data.complete_partial(x, out)
#         return out
    

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


class Classifier(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        output_dim = data.ydim - data.nknowns - data.neq

        layers = [nn.Linear(output_dim, 200), nn.LeakyReLU(), nn.Linear(200, 1), nn.Sigmoid()]
        # layers = [nn.Linear(output_dim+data.xdim, 200), nn.LeakyReLU(), nn.Linear(200, 1), nn.Sigmoid()]


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
    
    def __len__(self):
        return len(self.buffer_0) + len(self.buffer_1)

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
    
    def sample_0(self, batch_size):
        samples_0 = torch.stack(random.sample(self.buffer_0, min(len(self.buffer_0), batch_size)))
        labels_0 = torch.zeros(samples_0.shape[0], device=DEVICE)
        return samples_0, labels_0
    
    def sample_1(self, batch_size):
        samples_1 = torch.stack(random.sample(self.buffer_1, min(len(self.buffer_1), batch_size)))
        labels_1 = torch.zeros(samples_1.shape[0], device=DEVICE)
        return samples_1, labels_1


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            raise ValueError('Cannot lable can not be None')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

if __name__=='__main__':
    main()
