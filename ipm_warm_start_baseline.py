import os
import osqp
import time
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
import cyipopt as ipopt

from copy import deepcopy
from qpth.qp import QPFunction
from scipy.sparse import csc_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class General_QP(object):
    """
        minimize_x 0.5*x^T Q x + p^Tx
        s.t.       Gx <= c
                   Ax = b

    """
    def __init__(self, prob_type, learning_type, val_frac=0.0833, test_frac=0.0833, device='cuda:0', seed=17, **kwargs):
        super().__init__()

        self.device = device
        self.seed = seed
        self.learning_type = learning_type
        self.train_frac = 1 - val_frac - test_frac
        self.val_frac = val_frac
        self.test_frac = test_frac


        if prob_type == 'QP_DC3':
            file_path = kwargs['file_path']
            data = sio.loadmat(file_path)
            self.data_size = data['X'].shape[0]

            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size
            torch.manual_seed(self.seed)

            self.num_var = data['Q'].shape[0]
            self.num_ineq = data['G'].shape[0]
            self.num_eq = data['A'].shape[0]
            self.num_lb = 0
            self.num_ub = 0
        elif prob_type == 'QP':
            self.data_size = kwargs['data_size']
            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size
            torch.manual_seed(self.seed)

            self.num_var = kwargs['num_var']
            self.num_ineq = kwargs['num_ineq']
            self.num_eq = kwargs['num_eq']
            self.num_lb = 0
            self.num_ub = 0
        else:
            file_path = kwargs['file_path']
            data = sio.loadmat(file_path)
            self.data_size = data['Q'].shape[0]

            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size
            torch.manual_seed(self.seed)

            self.num_var = data['Q'].shape[1]
            try:
                self.num_ineq = data['G'].shape[1]
            except KeyError:
                self.num_ineq = 0

            try:
                self.num_eq = data['A'].shape[1]
            except KeyError:
                self.num_eq = 0

            try:
                self.num_lb = data['lb'].shape[1]
            except KeyError:
                self.num_lb = 0

            try:
                self.num_ub = data['ub'].shape[1]
            except KeyError:
                self.num_ub = 0


        if learning_type == 'train':
            if prob_type == 'QP_DC3':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.train_size, 1)
            elif prob_type == 'QP':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[:self.train_size]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[:self.train_size]
            else:
                self.Q = torch.tensor(data['Q'], device=self.device).float()[:self.train_size]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[:self.train_size]

            if self.num_eq != 0:
                if prob_type == 'QP_DC3':
                    self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                    self.b = torch.tensor(data['X'], device=self.device).float()[:self.train_size]
                elif prob_type == 'QP':
                    self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[:self.train_size]
                    self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[:self.train_size] - 1  # [-1, 1]
                else:
                    self.A = torch.tensor(data['A'], device=self.device).float()[:self.train_size]
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[:self.train_size]

            if self.num_ineq != 0:
                if prob_type == 'QP_DC3':
                    self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1,1)
                    self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.train_size, 1)
                elif prob_type == 'QP':
                    self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[:self.train_size]
                    self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2)
                else:
                    self.G = torch.tensor(data['G'], device=self.device).float()[:self.train_size]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[:self.train_size]

            if self.num_lb != 0:
                self.lb = torch.tensor(data['lb'], device=self.device).float()[:self.train_size]
            else:
                self.lb = -torch.inf
            if self.num_ub != 0:
                self.ub = torch.tensor(data['ub'], device=self.device).float()[:self.train_size]
            else:
                self.ub = torch.inf


        elif learning_type == 'val':
            if prob_type == 'QP_DC3':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.val_size, 1)
            elif prob_type == 'QP':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size:self.train_size + self.val_size]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
            else:
                self.Q = torch.tensor(data['Q'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
            if self.num_eq != 0:
                if prob_type == 'QP_DC3':
                    self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                    self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                elif prob_type == 'QP':
                    self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
                    self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[self.train_size:self.train_size + self.val_size] - 1  # [-1, 1]
                else:
                    self.A = torch.tensor(data['A'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
            if self.num_ineq != 0:
                if prob_type == 'QP_DC3':
                    self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1,1)
                    self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.val_size, 1)
                elif prob_type == 'QP':
                    self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
                    self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2)
                else:
                    self.G = torch.tensor(data['G'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
            if self.num_lb != 0:
                self.lb = torch.tensor(data['lb'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
            else:
                self.lb = -torch.inf
            if self.num_ub != 0:
                self.ub = torch.tensor(data['ub'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
            else:
                self.ub = torch.inf


        elif learning_type == 'test':
            if prob_type == 'QP_DC3':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.test_size, 1)
            elif prob_type == 'QP':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size + self.val_size:]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size + self.val_size:]
            else:
                self.Q = torch.tensor(data['Q'], device=self.device).float()[self.train_size + self.val_size:]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:]
            if self.num_eq != 0:
                if prob_type == 'QP_DC3':
                    self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1,1)
                    self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size + self.val_size:]
                elif prob_type == 'QP':
                    self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var),device=device)[self.train_size + self.val_size:]
                    self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[self.train_size + self.val_size:] - 1  # [-1, 1]
                else:
                    self.A = torch.tensor(data['A'], device=self.device).float()[self.train_size + self.val_size:]
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:]
            if self.num_ineq != 0:
                if prob_type == 'QP_DC3':
                    self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                    self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.test_size, 1)
                elif prob_type == 'QP':
                    self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var),
                                          device=device)[self.train_size + self.val_size:]
                    self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2)
                else:
                    self.G = torch.tensor(data['G'], device=self.device).float()[self.train_size + self.val_size:]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:]
            if self.num_lb != 0:
                self.lb = torch.tensor(data['lb'], device=self.device).float()[self.train_size + self.val_size:]
            else:
                self.lb = -torch.inf
            if self.num_ub != 0:
                self.ub = torch.tensor(data['ub'], device=self.device).float()[self.train_size + self.val_size:]
            else:
                self.ub = torch.inf


    def name(self):
        str = 'General_QP-{}-{}-{}-{}-{}'.format(self.num_var, self.num_ineq, self.num_eq, self.num_lb, self.num_ub)
        return str


    def obj_fn(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return 0.5 * torch.bmm(x.permute(0, 2, 1), torch.bmm(Q, x)) + torch.bmm(p.permute(0, 2, 1), x)

    def obj_grad(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return torch.bmm(Q, x) + p

    def ineq_resid(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.bmm(G, x) - c

    def ineq_dist(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.clamp(self.ineq_resid(x, G=G, c=c), 0)

    def eq_resid(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.bmm(A, x) - b

    def eq_dist(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.abs(self.eq_resid(x, A=A, b=b))

    def lower_bound_dist(self, x, **kwargs):
        lb = kwargs.get('lb', self.lb)
        return torch.clamp(lb - x, 0)

    def upper_bound_dist(self, x, **kwargs):
        ub = kwargs.get('ub', self.ub)
        return torch.clamp(x-ub, 0)

    def cal_kkt_info(self, x, eta, s, lamb, zl, zu, sigma, **kwargs):
        """
        x: [batch_size, num_var, 1]
        eta: [batch_size, num_ineq, 1]
        lamb: [batch_size, num_eq, 1]
        s: [batch_size, num_ineq, 1]
        zl: [batch_size, num_lb, 1]
        zu: [batch_size, num_ub, 1]

        return:
        H: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        r: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        mu: [batch_size, 1, 1]
        """
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        mu = 0
        if self.num_ineq != 0:
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
            mu += sigma * ((eta * s).sum(1).unsqueeze(-1))
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
            mu += sigma * ((zl * (x-lb)).sum(1).unsqueeze(-1))
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
            mu += sigma * ((zu * (ub-x)).sum(1).unsqueeze(-1))
        batch_size = Q.shape[0]
        # mu
        mu = mu/(self.num_ineq+self.num_lb+self.num_ub)

        # residual
        r_list = []
        r1 = torch.bmm(Q, x) + p
        if self.num_ineq != 0:
            r1 += torch.bmm(G.permute(0, 2, 1), eta)
        if self.num_eq != 0:
            r1 += torch.bmm(A.permute(0, 2, 1), lamb)
        if self.num_lb != 0:
            r1 += -zl
        if self.num_ub != 0:
            r1 += zu
        r_list.append(r1)

        if self.num_ineq != 0:
            r2 = torch.bmm(G, x) - c + s
            r3 = eta * s - mu
            r_list.append(r2)
            r_list.append(r3)

        if self.num_eq != 0:
            r4 = torch.bmm(A, x) - b
            r_list.append(r4)

        if self.num_lb != 0:
            r5 = zl*(x-lb) - mu
            r_list.append(r5)

        if self.num_ub != 0:
            r6 = zu*(ub-x) - mu
            r_list.append(r6)

        r = torch.concat(r_list, dim=1)

        # jacobian of residual
        H_list = []
        H1 = Q
        if self.num_ineq != 0:
            H1 = torch.concat((H1, G.permute(0,2,1)), dim=2)
        if self.num_eq != 0:
            H1 = torch.concat((H1, A.permute(0,2,1)), dim=2)
        if self.num_ineq != 0:
            H1 = torch.concat((H1, torch.zeros(size=(batch_size, self.num_var, self.num_ineq), device=self.device)), dim=2)
        if self.num_lb != 0:
            H1 = torch.concat((H1, -torch.diag_embed(torch.ones(size=(batch_size, self.num_lb), device=self.device))), dim=2)
        if self.num_ub != 0:
            H1 = torch.concat((H1, torch.diag_embed(torch.ones(size=(batch_size, self.num_ub), device=self.device))), dim=2)
        H_list.append(H1)

        if self.num_ineq != 0:
            H2 = torch.concat((G, torch.zeros(size=(batch_size, self.num_ineq, self.num_ineq), device=self.device)), dim=2)
            if self.num_eq != 0:
                H2 = torch.concat((H2, torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device)), dim=2)
            H2 = torch.concat((H2, torch.diag_embed(torch.ones(size=(batch_size, self.num_ineq), device=self.device))), dim=2)
            if self.num_lb != 0:
                H2 = torch.concat((H2, torch.zeros(size=(batch_size, self.num_ineq, self.num_lb), device=self.device)), dim=2)
            if self.num_ub != 0:
                H2 = torch.concat((H2, torch.zeros(size=(batch_size, self.num_ineq, self.num_ub), device=self.device)), dim=2)
            H_list.append(H2)

            H3 = torch.zeros(size=(batch_size, self.num_ineq, self.num_var), device=self.device)
            H3 = torch.concat((H3, torch.diag_embed(s.squeeze(-1))), dim=2)
            if self.num_eq != 0:
                H3 = torch.concat((H3, torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device)), dim=2)
            H3 = torch.concat((H3, torch.diag_embed(eta.squeeze(-1))), dim=2)
            if self.num_lb != 0:
                H3 = torch.concat((H3, torch.zeros(size=(batch_size, self.num_ineq, self.num_lb), device=self.device)), dim=2)
            if self.num_ub != 0:
                H3 = torch.concat((H3, torch.zeros(size=(batch_size, self.num_ineq, self.num_ub), device=self.device)), dim=2)
            H_list.append(H3)

        if self.num_eq != 0:
            H4 = A
            if self.num_ineq != 0:
                H4 = torch.concat((H4, torch.zeros(size=(batch_size, self.num_eq, self.num_ineq), device=self.device)), dim=2)
            H4 = torch.concat((H4, torch.zeros(size=(batch_size, self.num_eq, self.num_eq), device=self.device)), dim=2)
            if self.num_ineq != 0:
                H4 = torch.concat((H4, torch.zeros(size=(batch_size, self.num_eq, self.num_ineq), device=self.device)), dim=2)
            if self.num_lb != 0:
                H4 = torch.concat((H4, torch.zeros(size=(batch_size, self.num_eq, self.num_lb), device=self.device)), dim=2)
            if self.num_ub != 0:
                H4 = torch.concat((H4, torch.zeros(size=(batch_size, self.num_eq, self.num_ub), device=self.device)), dim=2)
            H_list.append(H4)

        if self.num_lb != 0:
            H5 = torch.diag_embed(zl.squeeze(-1))
            if self.num_ineq != 0:
                H5 = torch.concat((H5, torch.zeros(size=(batch_size, self.num_lb, self.num_ineq), device=self.device)), dim=2)
            if self.num_eq != 0:
                H5 = torch.concat((H5, torch.zeros(size=(batch_size, self.num_lb, self.num_eq), device=self.device)), dim=2)
            if self.num_ineq != 0:
                H5 = torch.concat((H5, torch.zeros(size=(batch_size, self.num_lb, self.num_ineq), device=self.device)), dim=2)
            H5 = torch.concat((H5, torch.diag_embed((x-lb).squeeze(-1))), dim=2)
            if self.num_ub != 0:
                H5 = torch.concat((H5, torch.zeros(size=(batch_size, self.num_lb, self.num_ub), device=self.device)), dim=2)
            H_list.append(H5)

        if self.num_ub != 0:
            H6 = -torch.diag_embed(zu.squeeze(-1))
            if self.num_ineq != 0:
                H6 = torch.concat((H6, torch.zeros(size=(batch_size, self.num_ub, self.num_ineq), device=self.device)), dim=2)
            if self.num_eq != 0:
                H6 = torch.concat((H6, torch.zeros(size=(batch_size, self.num_ub, self.num_eq), device=self.device)), dim=2)
            if self.num_ineq != 0:
                H6 = torch.concat((H6, torch.zeros(size=(batch_size, self.num_ub, self.num_ineq), device=self.device)), dim=2)
            if self.num_lb != 0:
                H6 = torch.concat((H6, torch.zeros(size=(batch_size, self.num_ub, self.num_lb), device=self.device)), dim=2)
            H6 = torch.concat((H6, torch.diag_embed((ub-x).squeeze(-1))), dim=2)
            H_list.append(H6)

        H = torch.concat(H_list, dim=1)
        return H, r, mu

    def sub_objective(self, y, H, r):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        H: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        r: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        1/2||H@delta_r-r||_2^2 = 1/2(deta_r^T@H^T@Hdelta_r)-deta_r^TH^Tr+1/2(r^Tr)
        """
        obj0 = 0.5 * torch.bmm(torch.bmm(y.permute(0, 2, 1), H.permute(0, 2, 1)), torch.bmm(H, y))
        obj1 = torch.bmm(torch.bmm(y.permute(0, 2, 1), H.permute(0, 2, 1)), r)
        obj2 = 0.5 * (torch.bmm(r.permute(0, 2, 1), r))
        return obj0+obj1+obj2

    def sub_smooth_grad(self, y, H, r):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        return H^T@H@delta_r+H^T@r
        """
        grad = torch.bmm(torch.bmm(H.permute(0, 2, 1), H), y) + torch.bmm(H.permute(0, 2, 1), r)
        return grad

    def opt_solve(self, solver_type='osqp', tol=1e-4, initial_y = None, init_mu=None, init_g=None, init_zl=None, init_zu=None):
        if solver_type == 'qpth':
            print('running qpth')
            iter_times = 0
            start_time = time.time()
            res = QPFunction(eps=tol, verbose=False)(self.Q, self.p, self.G, self.c, self.A, self.b)
            end_time = time.time()

            sols = np.array(res.detach().cpu().numpy())
            total_time = end_time - start_time
            parallel_time = total_time

        elif solver_type == 'osqp':
            print('running osqp')
            Q, p = self.Q.detach().cpu().numpy(), self.p.detach().cpu().numpy()
            if self.num_ineq != 0:
                G, c = self.G.detach().cpu().numpy(), self.c.detach().cpu().numpy()
            if self.num_eq != 0:
                A, b = self.A.detach().cpu().numpy(), self.b.detach().cpu().numpy()
            if self.num_lb != 0:
                lb = self.lb.detach().cpu().numpy()
            if self.num_ub != 0:
                ub = self.ub.detach().cpu().numpy()
            s = []
            iters = 0
            total_time = 0
            for i in range(Q.shape[0]):
                solver = osqp.OSQP()
                my_A = np.vstack([A[i, :, :], G[i, :, :]])
                my_l = np.hstack([b[i, :], -np.ones(c.shape[1]) * np.inf])
                my_u = np.hstack([b[i, :], c[i, :]])
                solver.setup(P=csc_matrix(Q[i, :, :]), q=p[i, :], A=csc_matrix(my_A),
                             l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += (end_time - start_time)
                if results.info.status == 'solved':
                    s.append(results.x)
                else:
                    s.append(np.ones(self.num_var) * np.nan)
                    print('Batch {} optimization failed.'.format(i))

            sols = np.array(s)
            parallel_time = total_time / Q.shape[0]

        elif solver_type == 'ipopt':
            Q, p = self.Q.detach().cpu().numpy(), self.p.detach().cpu().numpy()
            if self.num_ineq != 0:
                G, c = self.G.detach().cpu().numpy(), self.c.detach().cpu().numpy()
            if self.num_eq != 0:
                A, b = self.A.detach().cpu().numpy(), self.b.detach().cpu().numpy()
            if self.num_lb != 0:
                lb = self.lb.detach().cpu().numpy()
            else:
                lb = -np.infty * np.ones(shape=(Q.shape[0], Q.shape[1], 1))
            if self.num_ub != 0:
                ub = self.ub.detach().cpu().numpy()
            else:
                ub = np.infty * np.ones(shape=(Q.shape[0], Q.shape[1], 1))


            Y = []
            iters = []
            total_time = 0
            for i in range(Q.shape[0]):
                if initial_y is None:
                    # y0 = np.linalg.pinv(A[i]) @ b[i]  # feasible initial point
                    if (self.num_lb != 0) and (self.num_ub != 0):
                        y0 = ((lb[i]+ub[i])/2).squeeze(-1)
                    elif (self.num_lb != 0) and (self.num_ub == 0):
                        y0 = (lb[i] + np.ones(shape=lb[i].shape)).squeeze(-1)
                    elif (self.num_lb == 0) and (self.num_lb != 0):
                        y0 = (ub[i] - np.ones(shape=ub[i].shape)).squeeze(-1)
                    else:
                        y0 = np.zeros(self.num_var)
                else:
                    y0 = initial_y[i].cpu().numpy()

                # upper and lower bounds on constraints
                cls = []
                cus = []
                if self.num_ineq != 0:
                    cls.append(-np.inf * np.ones(G[i].shape[0]))
                    # cus.append(c[i].squeeze(-1))
                    cus.append(c[i])
                if self.num_eq != 0:
                    # cls.append(b[i].squeeze(-1))
                    # cus.append(b[i].squeeze(-1))
                    cls.append(b[i])
                    cus.append(b[i])
                cl = np.hstack(cls)
                cu = np.hstack(cus)

                if (self.num_ineq != 0) and (self.num_eq != 0):
                    G0, A0 = G[i], A[i]
                elif (self.num_ineq != 0) and (self.num_eq == 0):
                    G0, A0 = G[i], np.array(0.0)
                elif (self.num_ineq == 0) and (self.num_eq != 0):
                    G0, A0 = np.array(0.0), A[i]

                nlp = convex_ipopt(
                    Q[i],
                    # p[i].squeeze(-1),
                    p[i],
                    G0,
                    A0,
                    n=len(y0),
                    m=len(cl),
                    # problem_obj=prob_obj,
                    lb=lb[i],
                    ub=ub[i],
                    cl=cl,
                    cu=cu
                )

                nlp.add_option('tol', tol)
                nlp.add_option('print_level', 0)  # 3)
                if init_mu is not None:
                    nlp.add_option('warm_start_init_point', 'yes')
                    nlp.add_option('warm_start_bound_push', 1e-20)
                    nlp.add_option('warm_start_bound_frac', 1e-20)
                    nlp.add_option('warm_start_slack_bound_push', 1e-20)
                    nlp.add_option('warm_start_slack_bound_frac', 1e-20)
                    nlp.add_option('warm_start_mult_bound_push', 1e-20)
                    nlp.add_option('mu_strategy', 'monotone')
                    nlp.add_option('mu_init', init_mu[i].squeeze().cpu().item())

                start_time = time.time()
                if init_g is not None:
                    g = [x.item() for x in init_g[i].cpu()]
                else:
                    g = []

                if init_zl is not None:
                    zl = [x.item() for x in init_zl[i].cpu()]
                else:
                    zl = []

                if init_zu is not None:
                    zu = [x.item() for x in init_zu[i].cpu()]
                else:
                    zu = []

                y, info = nlp.solve(y0, lagrange=g, zl=zl, zu=zu)

                end_time = time.time()
                Y.append(y)
                iters.append(len(nlp.objectives))
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / Q.shape[0]
        else:
            raise NotImplementedError

        return sols, total_time, parallel_time, np.array(iters).mean()

    def warm_start_baseline(self, solver_type='ipopt', tol=1e-4, baseline_method="gauge", problem_type="Simple", num_var=100, num_ineq=50, num_eq=50, sol_path="/home/jxxiong/A-xjx/deeplde/baseline_sols/"):
        sol_path = os.path.join(sol_path, "_".join([baseline_method, problem_type, str(num_var), str(num_ineq), str(num_eq)]))
        Y0 = torch.tensor(sio.loadmat(sol_path)['x'], device=self.device, dtype=torch.float32)
        ineq_resid = torch.bmm(self.G, Y0.unsqueeze(-1)).squeeze(-1) - self.c
        eq_resid = torch.bmm(self.A, Y0.unsqueeze(-1)).squeeze(-1) - self.b
        print("========== Baseline Method: {} ==========".format(baseline_method))
        print('Initial ineq residual: ', ineq_resid.max().cpu().numpy())
        print('Initial eq residual: ', eq_resid.abs().max().cpu().numpy())
        sols, total_time, parallel_time, avg_iter = self.opt_solve(solver_type=solver_type, initial_y=Y0)
        sols = torch.tensor(sols, device=self.device, dtype=torch.float32)
        ineq_resid = torch.bmm(self.G, sols.unsqueeze(-1)).squeeze(-1) - self.c
        eq_resid = torch.bmm(self.A, sols.unsqueeze(-1)).squeeze(-1) - self.b
        print('Final ineq residual: ', ineq_resid.max().cpu().numpy())
        print('Final eq residual: ', eq_resid.abs().max().cpu().numpy())
        print('Total time: ', total_time)
        print('Parallel time: ', parallel_time)
        print('Average iteration: ', avg_iter)
        return total_time, parallel_time, avg_iter

class convex_ipopt(ipopt.Problem):
    def __init__(self, Q, p, G, A, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = Q
        self.p = p
        self.G = G
        self.A = A
        if (self.G == 0.0).all():
            self.num_ineq = 0
        else:
            self.num_ineq = self.G.shape[0]
        if (self.A == 0.0).all():
            self.num_eq = 0
        else:
            self.num_eq = self.A.shape[0]

        self.objectives = []
        self.mus = []
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p@y

    def gradient(self, y):
        return self.Q@y + self.p

    def constraints(self, y):
        const_values = []
        if self.num_ineq != 0:
            const_values.append(self.G@y)
        if self.num_eq != 0:
            const_values.append(self.A@y)
        return np.hstack(const_values)

    def jacobian(self, y):
        const_jacob = []
        if self.num_ineq != 0:
            const_jacob.append(self.G.flatten())
        if self.num_eq != 0:
            const_jacob.append(self.A.flatten())
        return np.concatenate(const_jacob)

    # # Don't use: In general, more efficient with numerical approx
    # def hessian(self, y, lagrange, obj_factor):
    #     H = obj_factor * (self.Q - np.diag(self.p * np.sin(y)) )
    #     return H[self.tril_indices]

    def intermediate(self, alg_mod, iter_count, obj_value,
            inf_pr, inf_du, mu, d_norm, regularization_size,
            alpha_du, alpha_pr, ls_trials):
        self.objectives.append(obj_value)
        self.mus.append(mu)


class Nonconvex_Op_DC3(object):
    """
        minimize_x 0.5*x^T Q x + p^Tsin(x)
        s.t.       Ax =  b
                   Gx <= c

    """
    def __init__(self, num_var, num_eq, num_ineq, data_size, learning_type, val_frac=0.0833, test_frac=0.0833, device='cpu', seed=17):
        super().__init__()
        self.num_var = num_var
        self.num_eq = num_eq
        self.num_ineq = num_ineq
        self.data_size = data_size
        self.device = device
        self.seed = seed
        self.learning_type = learning_type

        torch.manual_seed(self.seed)
        self.train_frac = 1 - val_frac - test_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.train_size = int(data_size * self.train_frac)
        self.val_size = int(data_size * val_frac)
        self.test_size = data_size - self.train_size - self.val_size

        filepath = os.path.join('datasets', 'nonconvex',
                                "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}.mat".format(self.num_var,
                                                                                          self.num_ineq,
                                                                                          self.num_eq,
                                                                                          self.data_size))
        data = sio.loadmat(filepath)

        if learning_type == 'train':
            self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
            self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.train_size, 1)
            self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
            self.b = torch.tensor(data['X'], device=self.device).float()[:self.train_size]
            self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
            self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.train_size, 1)

        elif learning_type == 'val':
            self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
            self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.val_size, 1)
            self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
            self.b = torch.tensor(data['X'], device=self.device).float()[
                     self.train_size:self.train_size + self.val_size]
            self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
            self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.val_size, 1)

            self.batch_size = self.val_size

        elif learning_type == 'test':
            self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
            self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.test_size, 1)
            self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
            self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size + self.val_size:]
            self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
            self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.test_size, 1)



    def name(self):
        str = 'Non-Convex Quadratic Pragramming-{}-{}-{}-{}'.format(self.batch_size,
                                                                self.num_var,
                                                                self.num_eq,
                                                                self.num_ineq)
        return str

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
    def b_np(self):
        return self.b.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def c_np(self):
        return self.c.detach().cpu().numpy()

    def obj_fn(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return 0.5*torch.bmm(x.permute(0, 2, 1), torch.bmm(Q, x))+torch.bmm(p.unsqueeze(-1).permute(0,2,1), torch.sin(x))

    def eq_resid(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.bmm(A, x) - b.unsqueeze(-1)

    def eq_dist(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.abs(self.eq_resid(x, A=A, b=b))

    def ineq_resid(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.bmm(G, x) - c.unsqueeze(-1)

    def ineq_dist(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.clamp(self.ineq_resid(x, G=G, c=c), 0)

    def cal_kkt_info(self, x, eta, lamb, s, sigma, **kwargs):
        """
        x: [batch_size, num_var, 1]
        eta: [batch_size, num_ineq, 1]
        lamb: [batch_size, num_eq, 1]
        s: [batch_size, num_ineq, 1]
        mu: [batch_size, 1]
        b: [batch_size, num_eq]

        return:
        r: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        H: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        """
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        batch_size = Q.shape[0]
        # mu
        mu = sigma * ((eta * s).sum(1).unsqueeze(-1)) / self.num_ineq

        # calculate KKT linear system residual vector
        r1 = torch.bmm(Q, x) + torch.bmm(torch.diag_embed(p), torch.cos(x)) + \
             torch.bmm(A.permute(0, 2, 1), lamb) + torch.bmm(G.permute(0, 2, 1), eta)
        r2 = torch.bmm(G, x) - c.unsqueeze(-1) + s
        r3 = eta * s - mu
        r4 = torch.bmm(A, x) - b.unsqueeze(-1)
        r = torch.concat((r1, r2, r3, r4), dim=1)

        # calculate KKT linear system matrix
        H1 = torch.concat((Q - torch.bmm(torch.diag_embed(p), torch.diag_embed(torch.sin(x).squeeze())),
                           G.permute(0, 2, 1), A.permute(0, 2, 1),
                           torch.zeros(size=(batch_size, self.num_var, self.num_ineq), device=self.device)), dim=2)
        H2 = torch.concat((G, torch.zeros(size=(batch_size, self.num_ineq, self.num_ineq), device=self.device),
                           torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device),
                           torch.diag_embed(torch.ones(size=(batch_size, self.num_ineq), device=self.device))), dim=2)
        H3 = torch.concat((torch.zeros(size=(batch_size, self.num_ineq, self.num_var), device=self.device),
                           torch.diag_embed(s.squeeze()),
                           torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device),
                           torch.diag_embed(eta.squeeze())), dim=2)
        H4 = torch.concat((A, torch.zeros(size=(batch_size, self.num_eq, self.num_ineq + self.num_ineq + self.num_eq), device=self.device)), dim=2)
        H = torch.concat((H1, H2, H3, H4), dim=1)
        return H, r, mu

    def sub_objective(self, y, H, r):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        H: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        r: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        1/2||H@delta_r-r||_2^2 = 1/2(deta_r^T@H^T@Hdelta_r)-deta_r^TH^Tr+1/2(r^Tr)
        """
        obj0 = 0.5 * torch.bmm(torch.bmm(y.permute(0, 2, 1), H.permute(0, 2, 1)), torch.bmm(H, y))
        obj1 = torch.bmm(torch.bmm(y.permute(0, 2, 1), H.permute(0, 2, 1)), r)
        obj2 = 0.5 * (torch.bmm(r.permute(0, 2, 1), r))
        return obj0+obj1+obj2

    def sub_smooth_grad(self, y, H, r):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        return H^T@H@delta_r+H^T@r
        """
        grad = torch.bmm(torch.bmm(H.permute(0, 2, 1), H), y) + torch.bmm(H.permute(0, 2, 1), r)
        return grad

    def opt_solve(self, solver_type='ipopt', tol=1e-4, initial_y=None, init_mu=None, init_g=None):
        Q = self.Q.detach().cpu().numpy()
        p = self.p.detach().cpu().numpy()
        G = self.G.detach().cpu().numpy()
        c = self.c.detach().cpu().numpy()
        A = self.A.detach().cpu().numpy()
        b = self.b.detach().cpu().numpy()

        Y = []
        iters = []
        total_time = 0
        if solver_type == 'ipopt':
            for i in range(Q.shape[0]):
                if initial_y is None:
                    # y0 = np.linalg.pinv(A[i]) @ b[i]  # feasible initial point
                    y0 = np.zeros(self.num_var)
                else:
                    y0 = initial_y[i].cpu().numpy()

                # upper and lower bounds on variables
                lb = -np.infty * np.ones(y0.shape)
                ub = np.infty * np.ones(y0.shape)

                # upper and lower bounds on constraints
                cl = np.hstack([-np.inf * np.ones(G[i].shape[0]), b[i]])
                cu = np.hstack([c[i], b[i]])

                nlp = nonconvex_ipopt(
                    Q[i],
                    p[i],
                    G[i],
                    A[i],
                    n=len(y0),
                    m=len(cl),
                    # problem_obj=prob_obj,
                    lb=lb,
                    ub=ub,
                    cl=cl,
                    cu=cu
                )
                nlp.add_option('tol', tol)
                nlp.add_option('print_level', 0)  # 3)

                if init_mu is not None:
                    nlp.add_option('warm_start_init_point', 'yes')
                    nlp.add_option('warm_start_bound_push', 1e-20)
                    nlp.add_option('warm_start_bound_frac', 1e-20)
                    nlp.add_option('warm_start_slack_bound_push', 1e-20)
                    nlp.add_option('warm_start_slack_bound_frac', 1e-20)
                    nlp.add_option('warm_start_mult_bound_push', 1e-20)
                    nlp.add_option('mu_strategy', 'monotone')
                    nlp.add_option('mu_init', init_mu[i].squeeze().cpu().item())
                start_time = time.time()

                if init_g is not None:
                    y, info = nlp.solve(y0, lagrange=[x.item() for x in init_g[i].cpu()])
                else:
                    y, info = nlp.solve(y0)
                end_time = time.time()
                Y.append(y)
                iters.append(len(nlp.objectives))
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / Q.shape[0]
        else:
            raise NotImplementedError

        return sols, total_time, parallel_time, np.array(iters).mean()
    
    def warm_start_baseline(self, solver_type='ipopt', tol=1e-4, baseline_method="gauge", problem_type="Simple", num_var=100, num_ineq=50, num_eq=50, sol_path="/home/jxxiong/A-xjx/deeplde/baseline_sols/"):
        sol_path = os.path.join(sol_path, "_".join([baseline_method, problem_type, str(num_var), str(num_ineq), str(num_eq)]))
        Y0 = torch.tensor(sio.loadmat(sol_path)['x'], device=self.device, dtype=torch.float32)
        ineq_resid = torch.bmm(self.G, Y0.unsqueeze(-1)).squeeze(-1) - self.c
        eq_resid = torch.bmm(self.A, Y0.unsqueeze(-1)).squeeze(-1) - self.b
        print("========== Baseline Method: {} ==========".format(baseline_method))
        print('Initial ineq residual: ', ineq_resid.max().cpu().numpy())
        print('Initial eq residual: ', eq_resid.abs().max().cpu().numpy())
        sols, total_time, parallel_time, avg_iter = self.opt_solve(solver_type=solver_type, initial_y=Y0)
        sols = torch.tensor(sols, device=self.device, dtype=torch.float32)
        ineq_resid = torch.bmm(self.G, sols.unsqueeze(-1)).squeeze(-1) - self.c
        eq_resid = torch.bmm(self.A, sols.unsqueeze(-1)).squeeze(-1) - self.b
        print('Final ineq residual: ', ineq_resid.max().cpu().numpy())
        print('Final eq residual: ', eq_resid.abs().max().cpu().numpy())
        print('Total time: ', total_time)
        print('Parallel time: ', parallel_time)
        print('Average iteration: ', avg_iter)
        return total_time, parallel_time, avg_iter

class nonconvex_ipopt(ipopt.Problem):
    def __init__(self, Q, p, G, A, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = Q
        self.p = p
        self.G = G
        self.A = A
        self.objectives = []
        self.mus = []
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p @ np.sin(y)

    def gradient(self, y):
        return self.Q @ y + (self.p * np.cos(y))

    def constraints(self, y):
        return np.hstack([self.G @ y, self.A @ y])

    def jacobian(self, y):
        return np.concatenate([self.G.flatten(), self.A.flatten()])

    # # Don't use: In general, more efficient with numerical approx
    # def hessian(self, y, lagrange, obj_factor):
    #     H = obj_factor * (self.Q - np.diag(self.p * np.sin(y)) )
    #     return H[self.tril_indices]

    def intermediate(self, alg_mod, iter_count, obj_value,
                    inf_pr, inf_du, mu, d_norm, regularization_size,
                    alpha_du, alpha_pr, ls_trials):
        self.objectives.append(obj_value)
        self.mus.append(mu)
        

if __name__ == "__main__":
    problem_type_list = ["SimpleProblem", "NonconvexProblem"]
    problem_size_list = [[100, 50, 50], [200, 100, 200]]
    num_example = 10000
    
    for problem_type in problem_type_list:
        for problem_size in problem_size_list:
            num_var = problem_size[0]
            num_ineq = problem_size[1]
            num_eq = problem_size[2]
            if problem_type == "SimpleProblem":    
                data = General_QP("QP_DC3", learning_type="test", file_path="/home/jxxiong/A-xjx/deeplde/datasets/simple/random_simple_dataset_var{}_ineq{}_eq{}_ex{}.mat".format(num_var, num_ineq, num_eq, num_example))
            elif problem_type == "NonconvexProblem":
                data = Nonconvex_Op_DC3(num_var=num_var, num_eq=num_eq, num_ineq=num_ineq, data_size=10000, learning_type="test")
            else:
                raise NotImplementedError
    
            total_time = []
            parallel_time = []
            avg_iter = []
            method_list = ["DC3", "deeplde", "EqH_Bis", "gauge"]
            for baseline_method in method_list:
                tt, pt, ai = data.warm_start_baseline(solver_type="ipopt", baseline_method=baseline_method, problem_type=problem_type, num_var=num_var, num_ineq=num_ineq, num_eq=num_eq)
                total_time.append(tt)
                parallel_time.append(pt)
                avg_iter.append(ai)

            # save the results as a dataframe
            df = pd.DataFrame({"Method": method_list, "Total Time": total_time, "Parallel Time": parallel_time, "Average Iteration": avg_iter})
            df.to_csv("/home/jxxiong/A-xjx/deeplde/baseline_sols/warmstart_{}_var{}_ineq{}_eq{}_ex{}.csv".format(problem_type, num_var, num_ineq, num_eq, num_example))                      
    