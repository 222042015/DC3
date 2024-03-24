import torch
from torch.autograd import Function
import numpy as np
import cvxpy as cp
import ipopt
import copy
import time

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)

###################################################################
# Base PROBLEM
###################################################################
class BaseProblem:
    def __init__(self, dataset, valid_frac=0.0833, test_frac=0.0833):
        self.input_L = torch.tensor(dataset['XL'] )
        self.input_U = torch.tensor(dataset['XU'] )
        self.L = torch.tensor(dataset['YL'] )
        self.U = torch.tensor(dataset['YU'] )
        self.X = torch.tensor(dataset['X'] )
        self.Y = torch.tensor(dataset['Y'] )
        self.num = dataset['X'].shape[0]
        self._device = DEVICE
        self.valid_frac = valid_frac
        self.test_frac = test_frac

    def eq_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            x = X[n].view(1, -1)
            y = Y[n].view(1, -1)
            y = torch.autograd.Variable(y, requires_grad=True)
            eq_penalty = self.eq_resid(x, y) ** 2
            eq_penalty = torch.sum(eq_penalty, dim=-1, keepdim=True)
            grad = torch.autograd.grad(eq_penalty, y)[0]
            grad_list.append(grad.view(1, -1))
        grad = torch.cat(grad_list, dim=0)
        return grad

    def ineq_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            x = X[n].view(1, -1)
            y = Y[n].view(1, -1)
            y = torch.autograd.Variable(y, requires_grad=True)
            ineq_penalty = self.ineq_resid(x, y) ** 2
            ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
            grad = torch.autograd.grad(ineq_penalty, y)[0]
            grad_list.append(grad.view(1, -1))
        grad = torch.cat(grad_list, dim=0)
        return grad

    def ineq_partial_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            Y_pred = Y[n, self.partial_vars].view(1, -1)
            x = X[n].view(1, -1)
            Y_pred = torch.autograd.Variable(Y_pred, requires_grad=True)
            y = self.complete_partial(x, Y_pred)
            # Y_comp = (x - Y_pred @ self.A_partial.T) @ self.A_other_inv.T
            # y = torch.zeros(1, self.ydim, device=X.device)
            # y[0, self.partial_vars] = Y_pred
            # y[0, self.other_vars] = Y_comp
            ineq_penalty = self.ineq_resid(x, y) ** 2
            ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
            grad_pred = torch.autograd.grad(ineq_penalty, Y_pred)[0]
            grad = torch.zeros(1, self.ydim, device=X.device)
            grad[0, self.partial_vars] = grad_pred
            grad[0, self.other_vars] = - (grad_pred @ self.A_partial.T) @ self.A_other_inv.T
            grad_list.append(grad)
        return torch.cat(grad_list, dim=0)

    def scale_full(self, X, Y):
        # lower_bound = self.L.view(1, -1)
        # upper_bound = self.U.view(1, -1)
        # The last layer of NN is sigmoid, scale to Opt bound
        scale_Y = Y * (self.U - self.L) + self.YL
        return scale_Y

    def scale_partial(self, X, Y):
        # lower_bound = (self.L[self.partial_vars]).view(1, -1)
        # upper_bound = (self.U[self.partial_vars]).view(1, -1)
        scale_Y = Y * (self.U - self.L) + self.L
        return scale_Y

    def scale(self, X, Y):
        if Y.shape[1] < self.ydim:
            Y_scale = self.scale_partial(X, Y)
        else:
            Y_scale = self.scale_full(X, Y)
        return Y_scale

    def cal_penalty(self, X, Y):
        penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
        return torch.abs(penalty)

    def check_feasibility(self, X, Y):
        return self.cal_penalty(X, Y)


###################################################################
# QP PROBLEM
###################################################################
class QPProblem(BaseProblem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
                   L<= x <=U
    """
    def __init__(self, dataset, valid_frac=0.0833, test_frac=0.0833):
        super().__init__(dataset, valid_frac=valid_frac, test_frac=test_frac)
        self.Q_np = dataset['Q']
        self.p_np = dataset['p']
        self.A_np = dataset['A']
        self.G_np = dataset['G']
        self.h_np = dataset['h']
        self.L_np = dataset['YL']
        self.U_np = dataset['YU']
        self.Q = torch.tensor(dataset['Q'] )
        self.p = torch.tensor(dataset['p'] )
        self.A = torch.tensor(dataset['A'] )
        self.G = torch.tensor(dataset['G'] )
        self.h = torch.tensor(dataset['h'] )
        self.L = torch.tensor(dataset['YL'] )
        self.U = torch.tensor(dataset['YU'] )
        self.X = torch.tensor(dataset['X'] )
        self.Y = torch.tensor(dataset['Y'] )
        self.xdim = dataset['X'].shape[1]
        self.ydim = dataset['Q'].shape[0]
        self.neq = dataset['A'].shape[0]
        self.nineq = dataset['G'].shape[0]
        self.nknowns = 0

        best_partial = dataset['best_partial']
        self.partial_vars = best_partial
        self.partial_unknown_vars = best_partial
        self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars)
        self.A_partial = self.A[:, self.partial_vars]
        self.A_other_inv = torch.inverse(self.A[:, self.other_vars])

        self.train_frac = 1 - valid_frac - test_frac
        self.valid_frac = valid_frac
        self.test_frac = test_frac
        self.trainX = self.X[:int(self.num*self.train_frac)]
        self.validX = self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]
        self.testX = self.X[int(self.num*(self.train_frac + self.valid_frac)):]

        self.trainY = self.Y[:int(self.num*self.train_frac)]
        self.validY = self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]
        self.testY = self.Y[int(self.num*(self.train_frac + self.valid_frac)):]


    def __str__(self):
        return 'QPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * Y).sum(dim=1)

    def eq_resid(self, X, Y):
        return Y @ self.A.T - X

    def complete_partial(self, X, Y, backward=True):
        Y_full = torch.zeros(X.shape[0], self.ydim, device=X.device)
        Y_full[:, self.partial_vars] = Y
        Y_full[:, self.other_vars] = (X - Y @ self.A_partial.T) @ self.A_other_inv.T
        return Y_full

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi in X_np:
                y = cp.Variable(self.ydim)
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                  [G @ y <= h, y <= U, y >= L,
                                   A @ y == Xi])
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return sols, total_time, parallel_time

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):

        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                                  [G @ y <= h, y <= U, y >= L,
                                   A @ y == Xi])
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return torch.tensor(sols )

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                y.value = y_pred
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                  [G @ y <= h, y <= U, y >= L,
                                   A @ y == Xi])
                start_time = time.time()
                prob.solve(warm_start=True)
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
        else:
            raise NotImplementedError
        return torch.tensor(sols )


###################################################################
# QCQP Problem
###################################################################
class QCQPProbem(QPProblem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   1/2 * y^T H y + G^T y <= h
                   L<= x <=U
    """
    def __init__(self, dataset, valid_frac=0.0833, test_frac=0.0833):
        super().__init__(dataset, valid_frac=valid_frac, test_frac=test_frac)
        self.H_np = dataset['H']
        self.H = torch.tensor(dataset['H'] )
        self._device = DEVICE

    def __str__(self):
        return 'QCQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def ineq_resid(self, X, Y):
        res = []
        """
         1/2 * y^T H y + G^T y <= h
         H: m * n * n
         G: m * n
         y: 1 * n
         h: 1 * m
        """
        q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
        q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
        res = 0.5 * q + torch.matmul(Y, self.G.T) - self.h
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return resids
    
    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi in X_np:
                y = cp.Variable(self.ydim)
                constraints = [A @ y == Xi, y <= U, y >= L]
                for i in range(self.nineq):
                    Ht = H[i]
                    Gt = G[i]
                    ht = h[i]
                    constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                  constraints)
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return sols, total_time, parallel_time

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):

        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                constraints = [A @ y == Xi, y <= U, y >= L]
                for i in range(self.nineq):
                    Ht = H[i]
                    Gt = G[i]
                    ht = h[i]
                    constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
                prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                                  constraints)
                start_time = time.time()
                prob.solve()
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return torch.tensor(sols )

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            Y = []
            total_time = 0
            n = 0
            for Xi, y_pred in zip(X_np, Y_pred):
                y = cp.Variable(self.ydim)
                y.value = y_pred
                constraints = [A @ y == Xi, y <= U, y >= L]
                for i in range(self.nineq):
                    Ht = H[i]
                    Gt = G[i]
                    ht = h[i]
                    constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                  constraints)
                start_time = time.time()
                prob.solve(warm_start=True)
                end_time = time.time()
                print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
                n += 1
                Y.append(y.value)
                total_time += (end_time - start_time)
            sols = np.array(Y)
            parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError
        return torch.tensor(sols )

