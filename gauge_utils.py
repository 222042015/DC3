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

from pypower.api import case57
from pypower.api import opf, makeYbus
from pypower import idx_bus, idx_gen, ppoption

import gurobipy as gp
from gurobipy import Model, GRB, QuadExpr, LinExpr

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

class SimpleProblem:
    """ 
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
    """
    def __init__(self, Q, p, A, G, h, X, L, U, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.cat([torch.tensor(G), torch.eye(G.shape[1]), -torch.eye(G.shape[1])], dim=0)
        self._h = torch.cat([torch.tensor(h), torch.tensor(U), -torch.tensor(L)], dim=0)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = G.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        i = 0
        while abs(det) < 0.0001 and i < 100:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._A[:, self._other_vars])
            i += 1
        if i == 100:
            raise Exception
        else:
            self._A_partial = self._A[:, self._partial_vars]
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])

        ### For Pytorch
        self._device = None
        self.G_tmp = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)

    def __str__(self):
        return 'SimpleProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

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
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

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
    
    @property
    def trainIP(self):
        return self.IP[:int(self.num*self.train_frac)]

    @property
    def validIP(self):
        return self.IP[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testIP(self):
        return self.IP[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self, Y):
        return (0.5*(Y@self.Q)*Y + self.p*Y).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_dist@self.G

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
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            for Xi in X_np:
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
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time/len(X_np)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def get_interior_point(self, X, M=1e4, tol=1e-4):
        # Q, p, A, G, h, L_partial, U_partial = self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np_partial, self.U_np_partial
        Q, p, A, G, h = self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np
        A_other_inv = self._A_other_inv.detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()
        tmp_h_coeff = G[:, self.other_vars] @A_other_inv
        tmp_G = G[:, self.partial_vars] - G[:, self.other_vars] @ A_other_inv @ A[:, self.partial_vars]

        IP = []
        for Xi in X_np:
            tmp_h = h - tmp_h_coeff @ Xi
            model = gp.Model("qp")
            model.setParam('OutputFlag', 0)
            model.setParam('FeasibilityTol', tol)

            n_vars = len(self.partial_vars)
            vars = []
            for i in range(n_vars):
                vars.append(model.addVar(lb=-np.infty, ub=np.infty, vtype=GRB.CONTINUOUS, name=f"yp_{i}"))

            ya = model.addVar(lb=-np.infty, ub=0, vtype=GRB.CONTINUOUS, name="ya")

            obj = LinExpr()
            obj.add(M * ya)
            model.setObjective(obj, GRB.MINIMIZE)

            for i in range(tmp_G.shape[0]):
                expr = LinExpr()
                for j in range(n_vars):
                    if tmp_G[i, j] != 0:
                        expr.add(vars[j] * tmp_G[i, j])
                model.addConstr(expr <= tmp_h[i] + ya)
            
            model.optimize()
            if model.status == GRB.Status.OPTIMAL and model.getVarByName("ya").x < 0:
                IP.append(np.array([v.x for v in vars]))
            else:
                IP.append(np.ones(n_vars) * np.nan)
        return np.array(IP)

    def remove_no_ip(self):
        IP = self.get_interior_point(self.X)
        feas_mask = ~np.isnan(IP).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(IP[feas_mask])
        self.IP = torch.tensor(IP[feas_mask])
        self.IP_np = IP[feas_mask]
        return IP

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)  
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y

    def gauge_unit_ball(self, v):
        # when the set is the l_infinity unit ball, the mapping is the l-infinity norm
        return torch.norm(v, p=float('inf'), dim=1)
    
    def get_tmp_h(self, X):
        return self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T

    def gauge_set(self, v, u0, X):
        tmp_h = self.get_tmp_h(X) - u0 @ self.G_tmp.T
        lhs = v @ self.G_tmp.T 
        return torch.max(lhs / tmp_h, dim=1)[0]
    
    def gauge_map(self, v, u0, X):
        phi_unit_ball = self.gauge_unit_ball(v)
        phi_set = self.gauge_set(v, u0, X)
        return (phi_unit_ball / phi_set).unsqueeze(1) * v + u0
        

class NonconvexProblem:
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
    """
    def __init__(self, Q, p, A, G, h, X, L, U, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        # self._G = torch.tensor(G)
        # self._h = torch.tensor(h)
        self._G = torch.cat([torch.tensor(G), torch.eye(G.shape[1]), -torch.eye(G.shape[1])], dim=0)
        self._h = torch.cat([torch.tensor(h), torch.tensor(U), -torch.tensor(L)], dim=0)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = G.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        i = 0
        while abs(det) < 0.0001 and i < 100:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._A[:, self._other_vars])
            i += 1
        if i == 100:
            raise Exception
        else:
            self._A_partial = self._A[:, self._partial_vars]
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])
            self._M = 2 * (self.G[:, self.partial_vars] -
                            self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial))

        ### For Pytorch
        self._device = None
        self.G_tmp = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)


    def __str__(self):
        return 'NonconvexProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

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
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

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
    
    @property
    def trainIP(self):
        return self.IP[:int(self.num*self.train_frac)]

    @property
    def validIP(self):
        return self.IP[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testIP(self):
        return self.IP[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self, Y):
        return (0.5*(Y@self.Q)*Y + self.p*torch.sin(Y)).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        return 2 * torch.clamp(Y@self.G.T - self.h, 0) @ self.G

    def ineq_partial_grad(self, X, Y):
        grad = torch.clamp(Y@self.G.T - self.h, 0) @ self._M
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

    def opt_solve(self, X, solver_type='ipopt', tol=1e-4):
        Q, p, A, G, h = self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np
        X_np = X.detach().cpu().numpy()
        Y = []
        total_time = 0
        for Xi in X_np:
            if solver_type == 'ipopt':
                y0 = np.linalg.pinv(A)@Xi  # feasible initial point

                # upper and lower bounds on variables
                lb = -np.infty * np.ones(y0.shape)
                ub = np.infty * np.ones(y0.shape)

                # upper and lower bounds on constraints
                cl = np.hstack([Xi, -np.inf * np.ones(G.shape[0])])
                cu = np.hstack([Xi, h])

                nlp = ipopt.problem(
                            n=len(y0),
                            m=len(cl),
                            problem_obj=nonconvex_ipopt(Q, p, A, G),
                            lb=lb,
                            ub=ub,
                            cl=cl,
                            cu=cu
                            )

                nlp.addOption('tol', tol)
                nlp.addOption('print_level', 0) # 3)

                start_time = time.time()
                y, info = nlp.solve(y0)
                end_time = time.time()
                Y.append(y)
                total_time += (end_time - start_time)
            else:
                raise NotImplementedError

        return np.array(Y), total_time, total_time/len(X_np)

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y
    
    def get_interior_point(self, X, M=1e4, tol=1e-4):
        Q, p, A, G, h = self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np
        A_other_inv = self._A_other_inv.detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()
        tmp_h_coeff = G[:, self.other_vars] @A_other_inv
        tmp_G = G[:, self.partial_vars] - G[:, self.other_vars] @ A_other_inv @ A[:, self.partial_vars]

        IP = []
        for Xi in X_np:
            tmp_h = h - tmp_h_coeff @ Xi
            model = gp.Model("qp")
            model.setParam('OutputFlag', 0)
            model.setParam('FeasibilityTol', tol)

            n_vars = len(self.partial_vars)
            vars = []
            for i in range(n_vars):
                vars.append(model.addVar(lb=-np.infty, ub=np.infty, vtype=GRB.CONTINUOUS, name=f"yp_{i}"))

            ya = model.addVar(lb=-np.infty, ub=0, vtype=GRB.CONTINUOUS, name="ya")

            obj = LinExpr()
            obj.add(M * ya)
            model.setObjective(obj, GRB.MINIMIZE)

            for i in range(tmp_G.shape[0]):
                expr = LinExpr()
                for j in range(n_vars):
                    if tmp_G[i, j] != 0:
                        expr.add(vars[j] * tmp_G[i, j])
                model.addConstr(expr <= tmp_h[i] + ya)
            
            model.optimize()
            if model.status == GRB.Status.OPTIMAL and model.getVarByName("ya").x < 0:
                IP.append(np.array([v.x for v in vars]))
            else:
                IP.append(np.ones(n_vars) * np.nan)
        return np.array(IP)

    def remove_no_ip(self):
        IP = self.get_interior_point(self.X)
        feas_mask = ~np.isnan(IP).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(IP[feas_mask])
        self.IP = torch.tensor(IP[feas_mask])
        self.IP_np = IP[feas_mask]
        return IP
    
    def gauge_unit_ball(self, v):
        # when the set is the l_infinity unit ball, the mapping is the l-infinity norm
        return torch.norm(v, p=float('inf'), dim=1)
    
    def get_tmp_h(self, X):
        return self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T

    def gauge_set(self, v, u0, X):
        tmp_h = self.get_tmp_h(X) - u0 @ self.G_tmp.T
        lhs = v @ self.G_tmp.T 
        return torch.max(lhs / tmp_h, dim=1)[0]
    
    def gauge_map(self, v, u0, X):
        phi_unit_ball = self.gauge_unit_ball(v)
        phi_set = self.gauge_set(v, u0, X)
        return (phi_unit_ball / phi_set).unsqueeze(1) * v + u0

class nonconvex_ipopt(object):
    def __init__(self, Q, p, A, G):
        self.Q = Q
        self.p = p
        self.A = A
        self.G = G
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p@np.sin(y)

    def gradient(self, y):
        return self.Q@y + (self.p * np.cos(y))

    def constraints(self, y):
        return np.hstack([self.A@y, self.G@y])

    def jacobian(self, y):
        return np.concatenate([self.A.flatten(), self.G.flatten()])