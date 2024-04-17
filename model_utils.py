import torch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NNSolver_eq_proj(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        # layer_sizes = [data.xdim, 1024, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], data.ydim)]
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        prob_type = self._args['probType']
        if prob_type == 'simple':
            return self.net(x)
        elif prob_type == 'nonconvex':
            return self.net(x)
        elif prob_type == 'convex_qcqp':
            return self.net(x)
        elif 'acopf' in prob_type:
            out = self.net(x)
            data = self._data
            out2 = nn.Sigmoid()(out[:, :-data.nbus])
            pg = out2[:, :data.ng] * data.pmax + (1-out2[:, :data.ng]) * data.pmin
            qg = out2[:, data.ng:2*data.ng] * data.qmax + (1-out2[:, data.ng:2*data.ng]) * data.qmin
            vm = out2[:, 2*data.ng:] * data.vmax + (1- out2[:, 2*data.ng:]) * data.vmin
            return torch.cat([pg, qg, vm, out[:, -data.nbus:]], dim=1)
        elif 'dcopf' in prob_type:
            out = self.net(x)
            data = self._data
            out[:, data.bounded_index] = nn.functional.sigmoid((out[:, data.bounded_index])) * data.Ub[data.bounded_index] + (1 - nn.functional.sigmoid((out[:, data.bounded_index]))) * data.Lb[data.bounded_index]
            out[:, data.ub_only_index] = -nn.functional.relu(out[:, data.ub_only_index]) + data.Ub[data.ub_only_index] 
            # out[:, data.lb_only_index] = nn.functional.relu(out[:, data.lb_only_index] - self.Lb[data.lb_only_index]) + self.Lb[data.lb_only_index]
            out[:, data.lb_only_index] = nn.functional.relu(out[:, data.lb_only_index]) + data.Lb[data.lb_only_index]

            return out
        else:
            raise NotImplementedError


class NNSolver_DC3(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
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

            return self._data.complete_partial(x, out)
        else:
            return self._data.process_output(x, out)


class NNSolver_baseline_nn(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], data.ydim)]
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)
        
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

class NNSolver_baseline_eq_nn(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], data.ydim - data.nknowns - data.neq)]
        if 'acopf' in self._args['probType']:
            layers += [nn.Sigmoid()]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        prob_type = self._args['probType']
        if prob_type in ['simple', 'nonconvex']:
            return self.net(x)
        elif 'acopf' in prob_type:
            Z = self.net(x)
            data = self._data

            Z_scaled = torch.zeros(Z.shape, device=DEVICE)

            # Re-scale real powers
            Z_scaled[:, data.pg_pv_zidx] = Z[:, data.pg_pv_zidx] * data.pmax[1:] + \
                 (1-Z[:, data.pg_pv_zidx]) * data.pmin[1:]
            
            # Re-scale voltage magnitudes
            Z_scaled[:, data.vm_spv_zidx] = Z[:, data.vm_spv_zidx] * data.vmax[data.spv] + \
                (1-Z[:, data.vm_spv_zidx]) * data.vmin[data.spv]

            return Z_scaled