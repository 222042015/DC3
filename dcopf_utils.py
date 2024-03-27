import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)

import numpy as np
import osqp
from qpth.qp import QPFunction
# import ipopt
import cyipopt as ipopt
from scipy.linalg import svd
from scipy.sparse import csc_matrix

import hashlib
from copy import deepcopy
import scipy.io as spio
import time

from pypower.api import case57, case39, case118, case300
from pypower.api import opf, makeYbus, ext2int
from pypower import idx_bus, idx_gen, ppoption

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def my_hash(string):
    return hashlib.sha1(bytes(string, 'utf-8')).hexdigest()


###################################################################
# SIMPLE PROBLEM
###################################################################

class DcopfProblem:
    """ 
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
                   l <= y <= u
    """
    # def __init__(self, Q, p, A, G, h, X, Lb, Ub, valid_frac=0.0833, test_frac=0.0833):
    def __init__(self, dataset, valid_frac=0.0833, test_frac=0.0833):
        A = np.array(dataset['A'])
        X = np.array(dataset['X'])
        Y = np.array(dataset['Y'])
        G = np.array(dataset['G'])
        h = np.array(dataset['h'])
        Q = np.array(dataset['Q'])
        p = np.array(dataset['p'])
        Lb = np.array(dataset['Lb'])
        Ub = np.array(dataset['Ub'])
        self.Q = torch.tensor(Q)
        self.p = torch.tensor(p)
        self.A = torch.tensor(A)
        self.G = torch.tensor(G)
        self.h = torch.tensor(h)
        self.X = torch.tensor(X)
        self.Y = torch.tensor([0])
        self.Lb = torch.tensor(Lb)
        self.Ub = torch.tensor(Ub)
        self.xdim = X.shape[1]
        self.ydim = Q.shape[0]
        self.num = X.shape[0]
        self.neq = A.shape[0]
        self.nineq = G.shape[0]
        self.nknowns = 0
        self.valid_frac = valid_frac
        self.test_frac = test_frac

        lb_index = np.where(self.Lb != -np.inf)[0]
        ub_index = np.where(self.Ub != np.inf)[0]
        self.lb_index = lb_index
        self.ub_index = ub_index
        self.bounded_index = np.intersect1d(lb_index, ub_index)


        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'SimpleProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    def obj_fn(self, Y):
        return (0.5*(Y@self.Q)*Y + self.p*Y).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        ineq = Y@self.G.T - self.h
        # lb_violation = torch.maximum(torch.zeros(self.ydim), self.Lb - Y)

        # Upper bound violation: max(0, y - ub)
        # ub_violation = torch.maximum(torch.zeros(self.ydim), Y - self.Ub)
        lb_violation = self.Lb[self.lb_index] - Y[:, self.lb_index]
        ub_violation = Y[:, self.ub_index] - self.Ub[self.ub_index]

        # lb_violation = self.Lb - Y
        return torch.cat((ineq, lb_violation, ub_violation), dim=1)

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        # return 2*ineq_dist@self.G
        return torch.cat([2*ineq_dist@self.G, torch.eye(self.ydim), -torch.eye(self.ydim)], dim=0)

    def ineq_partial_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    def opt_solve(self, X, solver_type='osqp', tol=1e-4):

        if solver_type == 'qpth':
            print('running qpth')
            start_time = time.time()
            res = QPFunction(eps=tol, verbose=False)(self.Q, self.p, self.G, self.h, self.A, X)
            end_time = time.time()

            sols = np.array(res.detach().cpu().numpy())
            total_time = end_time - start_time
            parallel_time = total_time
        
        elif solver_type == 'osqp':
            print('running osqp')
            Q, p, A, G, h = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            c = 0
            for i, Xi in enumerate(X_np):
                solver = osqp.OSQP()
                my_A = np.vstack([A, G])
                my_l = np.hstack([Xi, -np.ones(h.shape[0]) * np.inf])
                my_u = np.hstack([Xi, h])
                solver.setup(P=csc_matrix(Q), q=p, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += (end_time - start_time)
                if results.info.status == 'solved':
                    Y.append(results.x)
                    c += 1
                else:
                    Y.append(np.ones(self.ydim) * np.nan)
                print(f"{i} / {c}")

            sols = np.array(Y)
            parallel_time = total_time/len(X_np)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)  
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y