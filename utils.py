import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)

import numpy as np
import osqp
from qpth.qp import QPFunction
import ipopt
from scipy.linalg import svd
from scipy.sparse import csc_matrix

import hashlib
from copy import deepcopy
import scipy.io as spio
import time

from pypower.api import case57, case300
from pypower.api import opf, makeYbus, savecase, ext2int, makeSbus, newtonpf
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
# ACOPF
###################################################################


CASE_FNS = dict([(57, case57)])

class ACOPFProblem:
    """
        minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
        s.t.                  p_g min   <= p_g  <= p_g max
                              q_g min   <= q_g  <= q_g max
                              vmag min  <= vmag <= vmag max
                              vang_slack = \theta_slack   # voltage angle     
                              (p_g - p_d) + (q_g - q_d)i = diag(vmag e^{i*vang}) conj(Y) (vmag e^{-i*vang})
    """

    def __init__(self, data, train_num=1000, valid_num=100, test_num=100):
        self.sample = data['sample']
        ppc = data['ppc']
        self.ppc = ppc
        # reset the bus index to start from 0
        self.ppc['bus'][:, idx_bus.BUS_I] -= 1
        self.ppc['gen'][:, idx_gen.GEN_BUS] -= 1
        self.ppc['branch'][:, [0, 1]] -= 1

        self.genbase = ppc['gen'][:, idx_gen.MBASE]
        self.baseMVA = ppc['baseMVA']

        demand = data['Dem'] / self.baseMVA
        gen = data['Gen'] / self.genbase
        self.voltage = data['Vol']

        self.data = data

        self.nbus = self.voltage.shape[1]

        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))

        # indices within gens
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.ng = ppc['gen'].shape[0]
        self.nb = ppc['bus'].shape[0]
        self.nslack = len(self.slack)
        self.npv = len(self.pv)
        assert self.nb == self.nbus

        self.quad_costs = torch.tensor(ppc['gencost'][:,4], dtype=torch.get_default_dtype())
        self.lin_costs  = torch.tensor(ppc['gencost'][:,5], dtype=torch.get_default_dtype())
        self.const_cost = ppc['gencost'][:,6].sum()

        self.pmax = torch.tensor(ppc['gen'][:,idx_gen.PMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(ppc['gen'][:,idx_gen.PMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(ppc['gen'][:,idx_gen.QMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(ppc['gen'][:,idx_gen.QMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(ppc['bus'][:,idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(ppc['bus'][:,idx_bus.VMIN], dtype=torch.get_default_dtype())
        self.slackva = torch.tensor([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])], 
            dtype=torch.get_default_dtype()).squeeze(-1)

        # retrieve the Ybus matrix from the input data
        self.Ybus = data['Ybus']
        self.Ybus = self.Ybus.todense()
        self.Ybusr = torch.tensor(np.real(self.Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(self.Ybus), dtype=torch.get_default_dtype())

        X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)
        Y = np.concatenate([np.real(gen), np.imag(gen), np.abs(self.voltage), np.angle(self.voltage)], axis=1)
        feas_mask =  ~np.isnan(Y).any(axis=1)

        self._X = torch.tensor(X[feas_mask], dtype=torch.get_default_dtype())
        self._Y = torch.tensor(Y[feas_mask], dtype=torch.get_default_dtype())
        self._xdim = X.shape[1]
        self._ydim = Y.shape[1]
        self._num = feas_mask.sum()

        self._neq = 2*self.nbus
        self._nineq = 4*self.ng + 2*self.nbus
        self._nknowns = self.nslack

        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2*self.ng
        self.va_start_yidx = 2*self.ng + self.nbus


        ## Keep parameters indicating how data was generated
        self.EPS_INTERIOR = data['EPS_INTERIOR'][0][0]
        self.CorrCoeff = data['CorrCoeff'][0][0]
        self.MaxChangeLoad = data['MaxChangeLoad'][0][0]


        ## Define train/valid/test split
        # self._valid_frac = valid_frac
        # self._test_frac = test_frac
        self._train_num = train_num
        self._valid_num = valid_num
        self._test_num = test_num
        assert self.train_num + self.valid_num + self.test_num <= self._num

        ## Define variables and indices for "partial completion" neural network

        # pg (non-slack) and |v|_g (including slack)
        self._partial_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv, self.va_start_yidx + self.slack])
        self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)
        self._partial_unknown_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv])

        # initial values for solver
        self.pg_init = torch.tensor(ppc['gen'][:, idx_gen.PG] / self.genbase)
        self.qg_init = torch.tensor(ppc['gen'][:, idx_gen.QG] / self.genbase)
        self.vm_init = torch.tensor(ppc['bus'][:, idx_bus.VM])
        self.va_init = torch.tensor(np.deg2rad(ppc['bus'][:, idx_bus.VA]))

        # voltage angle at slack buses (known)
        self.slack_va = self.va_init[self.slack]

        # indices of useful quantities in partial solution
        self.pg_pv_zidx = np.arange(self.npv)
        self.vm_spv_zidx = np.arange(self.npv, 2*self.npv + self.nslack)

        # useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus


        ### For Pytorch
        self._device = None
        # print(self.eq_resid(self.X[0].unsqueeze(0), self.Y[0].unsqueeze(0)).abs().max())
        # print(self.eq_resid2(self.X[0].unsqueeze(0), self.Y[0].unsqueeze(0)).abs().max())


    def __str__(self):
        return 'ACOPF-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            self.nbus,
            self.sample,
            self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,
            self.train_num, self.valid_num, self.test_num)

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
        return self._partial_unknown_vars

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
    def valid_num(self):
        return self._valid_num

    @property
    def test_num(self):
        return self._test_num

    @property
    def train_num(self):
        return self._train_num

    @property
    def trainX(self):
        return self.X[:self.train_num]
    @property
    def validX(self):
        return self.X[self.train_num:self.train_num+self.valid_num]

    @property
    def testX(self):
        return self.X[self.train_num+self.valid_num:self.train_num+self.valid_num+self.test_num]

    @property
    def trainY(self):
        return self.Y[:self.train_num]
    @property
    def validY(self):
        return self.Y[self.train_num:self.train_num+self.valid_num]

    @property
    def testY(self):
        return self.Y[self.train_num+self.valid_num:self.train_num+self.valid_num+self.test_num]

    @property
    def device(self):
        return self._device

    def get_yvars(self, Y):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2*self.ng]
        vm = Y[:, -2*self.nbus:-self.nbus]
        va = Y[:, -self.nbus:]
        return pg, qg, vm, va

    def obj_fn(self, Y):
        pg, _, _, _ = self.get_yvars(Y)
        pg_mw = pg * torch.tensor(self.genbase).to(self.device)
        cost = (self.quad_costs * pg_mw**2).sum(axis=1) + \
            (self.lin_costs * pg_mw).sum(axis=1) + \
            self.const_cost
        return cost / (self.genbase.mean() ** 2)

    def eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        tmp1 = torch.squeeze(torch.matmul(self.Ybusr, vr.unsqueeze(-1)) - torch.matmul(self.Ybusi, vi.unsqueeze(-1)))
        tmp2 = -torch.squeeze(torch.matmul(self.Ybusi, vr.unsqueeze(-1)) + torch.matmul(self.Ybusr, vi.unsqueeze(-1)))

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self.device)
        pg_expand[:, self.spv] = pg 
        real_resid = (pg_expand - X[:, :self.nbus]) - (vr*tmp1 - vi*tmp2)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self.device)
        qg_expand[:, self.spv] = qg 
        react_resid = (qg_expand - X[:, self.nbus:]) - (vr*tmp2 + vi*tmp1)

        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)
        return resids

    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        resids = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            vm - self.vmax,
            self.vmin - vm
        ], dim=1)
        return resids

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        eq_resid = self.eq_resid(X,Y)
        return 2*eq_jac.transpose(1,2).bmm(eq_resid.unsqueeze(-1)).squeeze(-1)

    def ineq_grad(self, X, Y):
        ineq_jac = self.ineq_jac(Y)
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_jac.transpose(1,2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)

    def ineq_partial_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.linalg.solve(eq_jac[:, :, self.other_vars], eq_jac[:, :, self.partial_vars])

        direct_grad = self.ineq_grad(X, Y)
        indirect_partial_grad = dynz_dz.transpose(1,2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars]

        full_grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
        full_grad[:, self.partial_vars] = full_partial_grad
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)

        return full_grad

    def eq_jac(self, Y):
        _, _, vm, va = self.get_yvars(Y)

        # helper functions
        mdiag = lambda v1, v2: torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        Ydiagv = lambda Y, v: Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        dtm = lambda v, M: torch.diag_embed(v).bmm(M)

        # helper quantities
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        Yr = self.Ybusr
        Yi = self.Ybusi

        YrvrYivi = torch.squeeze(torch.matmul(self.Ybusr, vr.unsqueeze(-1)) - torch.matmul(self.Ybusi, vi.unsqueeze(-1)))
        YivrYrvi = torch.squeeze(torch.matmul(self.Ybusi, vr.unsqueeze(-1)) + torch.matmul(self.Ybusr, vi.unsqueeze(-1)))

        # real power equations
        dreal_dpg = torch.zeros(self.nbus, self.ng, device=self.device) 
        dreal_dpg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreal_dvm = -mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva)) \
            -mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva))
        dreal_dva = -mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr)) \
            -mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr))
        
        # reactive power equations
        dreact_dqg = torch.zeros(self.nbus, self.ng, device=self.device)
        dreact_dqg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreact_dvm = mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva)) \
            -mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva))
        dreact_dva = mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr)) \
            -mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr))

        jac = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape), 
                torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreal_dvm, dreal_dva], dim=2),
            torch.cat([torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreact_dqg.unsqueeze(0).expand(vr.shape[0], *dreact_dqg.shape),
                dreact_dvm, dreact_dva], dim=2)],
            dim=1)

        return jac


    def ineq_jac(self, Y):
        jac = torch.cat([
            torch.cat([torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([-torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device),
                torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device), 
                -torch.eye(self.ng, device=self.device),
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device),
                torch.zeros(self.nbus, self.ng, device=self.device), 
                torch.eye(self.nbus, device=self.device), 
                torch.zeros(self.nbus, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device), 
                torch.zeros(self.nbus, self.ng, device=self.device),
                -torch.eye(self.nbus, device=self.device), 
                torch.zeros(self.nbus, self.nbus, device=self.device)], dim=1)
            ], dim=0)
        return jac.unsqueeze(0).expand(Y.shape[0], *jac.shape)

    # Processes intermediate neural network output
    def process_output(self, X, out):
        out2 = nn.Sigmoid()(out[:, :-self.nbus+self.nslack])
        pg = out2[:, :self.qg_start_yidx] * self.pmax + (1-out2[:, :self.qg_start_yidx]) * self.pmin
        qg = out2[:, self.qg_start_yidx:self.vm_start_yidx] * self.qmax + \
            (1-out2[:, self.qg_start_yidx:self.vm_start_yidx]) * self.qmin
        vm = out2[:, self.vm_start_yidx:] * self.vmax + (1- out2[:, self.vm_start_yidx:]) * self.vmin

        va = torch.zeros(X.shape[0], self.nbus, device=self.device)
        va[:, self.nonslack_idxes] = out[:, self.va_start_yidx:]
        va[:, self.slack] = torch.tensor(self.slack_va, device=self.device).unsqueeze(0).expand(X.shape[0], self.nslack)

        return torch.cat([pg, qg, vm, va], dim=1)

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y_partial = torch.zeros(Z.shape, device=self.device)

        Y_partial[:, self.pg_pv_zidx] = Z[:, self.pg_pv_zidx] * (self.pmax[self.pv_] - self.pmin[self.pv_]) + self.pmin[
            self.pv_]
        # Re-scale voltage magnitudes
        Y_partial[:, self.vm_spv_zidx] = Z[:, self.vm_spv_zidx] * (self.vmax[self.spv] - self.vmin[self.spv]) + \
                                        self.vmin[self.spv]

        return PFFunction(self)(X, Y_partial)[0]

    def complete_partial2(self, X, Z):
        return PFFunction(self)(X, Z)


    def opt_solve(self, X, solver_type='pypower', tol=1e-6):
        X_np = X.detach().cpu().numpy()
        ppc = self.ppc

        # Set reduced voltage bounds if applicable
        ppc['bus'][:,idx_bus.VMIN] = ppc['bus'][:,idx_bus.VMIN] + self.EPS_INTERIOR
        ppc['bus'][:,idx_bus.VMAX] = ppc['bus'][:,idx_bus.VMAX] - self.EPS_INTERIOR

        # Solver options
        ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol)  # MIPS PDIPM

        Y = []
        total_time = 0
        for i in range(X_np.shape[0]):
            print(i)
            ppc['bus'][:, idx_bus.PD] = X_np[i, :self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = X_np[i, self.nbus:] * self.baseMVA

            start_time = time.time()
            my_result = opf(ppc, ppopt)
            print(my_result["success"])
            # printpf(my_result)
            end_time = time.time()
            total_time += (end_time - start_time)

            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))

        return np.array(Y), total_time, total_time/len(X_np)

def PFFunction(data, tol=1e-5, bsz=200, max_iters=20):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            # start_time = time.time()
            ## Step 1: Newton's method
            Y = torch.zeros(X.shape[0], data.ydim, device=X.device)
            # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
            Y[:, data.pg_start_yidx + data.pv_] = Z[:, data.pg_pv_zidx]  # pg at non-slack gens
            Y[:, data.vm_start_yidx + data.spv] = Z[:, data.vm_spv_zidx]  # vm at gens
            # init guesses for remaining values
            Y[:, data.vm_start_yidx + data.pq] = data.vm_init[data.pq]  # vm at load buses
            Y[:, data.va_start_yidx: data.va_start_yidx+data.nb] = data.va_init   # va at all bus
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = 0  # qg at gens (not used in Newton upd)
            Y[:, data.pg_start_yidx + data.slack_] = 0  # pg at slack (not used in Newton upd)

            keep_constr = np.concatenate([
                data.pflow_start_eqidx + data.pv,  # real power flow at non-slack gens
                data.pflow_start_eqidx + data.pq,  # real power flow at load buses
                data.qflow_start_eqidx + data.pq])  # reactive power flow at load buses
            newton_guess_inds = np.concatenate([
                data.vm_start_yidx + data.pq,  # vm at load buses
                data.va_start_yidx + data.pv,  # va at non-slack gens
                data.va_start_yidx + data.pq])  # va at load buses

            converged = torch.zeros(X.shape[0])
            jacs = []
            # newton_jacs_inv = []
            for b in range(0, X.shape[0], bsz):
                X_b = X[b:b + bsz]
                Y_b = Y[b:b + bsz]
                for _ in range(max_iters):
                    gy = data.eq_resid(X_b, Y_b)[:, keep_constr]
                    jac_full = data.eq_jac(Y_b)
                    jac = jac_full[:, keep_constr, :]
                    jac = jac[:, :, newton_guess_inds]

                    """Linear system"""
                    try:
                        delta = torch.linalg.solve(jac, gy.unsqueeze(-1)).squeeze(-1)
                    except:
                        # add small value to diagonal to ensure invertibility
                        jac += 1e-5*torch.eye(jac.shape[1], device=jac.device).unsqueeze(0).expand(jac.shape[0], *jac.shape[1:])
                        delta = torch.linalg.solve(jac, gy.unsqueeze(-1)).squeeze(-1)

                    Y_b[:, newton_guess_inds] -= delta
                    if torch.abs(delta).max() < tol:
                        break
                if torch.abs(delta).max() > tol:
                    print('Newton methods for Power Flow does not converge')
                # print(torch.abs(delta).max())
                converged[b:b + bsz] = (delta.abs() < tol).all(dim=1)
                jacs.append(jac_full)
                # newton_jacs_inv.append(newton_jac_inv)
                if (converged[b:b+bsz] == 0).sum() > 0:
                    print("number of non-converged samples: {}".format((converged[b:b+bsz] == 0).sum()))

            ## Step 2: Solve for remaining variables
            # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.qflow_start_eqidx + data.spv]
            # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
            Y[:, data.pg_start_yidx + data.slack_] = \
                -data.eq_resid(X, Y)[:, data.pflow_start_eqidx + data.slack]

            ctx.data = data
            ctx.save_for_backward(torch.cat(jacs),
                                  torch.as_tensor(newton_guess_inds, device=X.device),
                                  torch.as_tensor(keep_constr, device=X.device))
            return Y, converged

        @staticmethod
        def backward(ctx, dl_dy, dl_dc):

            data = ctx.data
            # jac, newton_jac_inv, newton_guess_inds, keep_constr = ctx.saved_tensors
            jac, newton_guess_inds, keep_constr = ctx.saved_tensors

            ## Step 2 (calc pg at slack and qg at gens)
            jac_pre_inv = jac[:, keep_constr, :]
            jac_pre_inv = jac_pre_inv[:, :, newton_guess_inds]

            # gradient of all voltages through step 3 outputs
            last_eqs = np.concatenate([data.pflow_start_eqidx + data.slack, data.qflow_start_eqidx + data.spv])
            last_vars = np.concatenate([
                data.pg_start_yidx + data.slack_, np.arange(data.qg_start_yidx, data.qg_start_yidx + data.ng)])
            # last_vars = np.concatenate([
            #     data.pg_start_yidx + data.slack_, data.qg_start_yidx + data.spv])
            jac3 = jac[:, last_eqs, :]
            dl_dvmva_3 = -jac3[:, :, data.vm_start_yidx:].transpose(1, 2).bmm( # dl/dz2*dz2/dz1
                dl_dy[:, last_vars].unsqueeze(-1)).squeeze(-1)

            # gradient of pd at slack and qd at gens through step 3 outputs
            dl_dpdqd_3 = dl_dy[:, last_vars] 

            # insert into correct places in x and y loss vectors
            dl_dy_3 = torch.zeros(dl_dy.shape, device=jac.device)
            dl_dy_3[:, data.vm_start_yidx:] = dl_dvmva_3

            dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=jac.device)
            dl_dx_3[:, np.concatenate([data.slack, data.nbus + data.spv])] = dl_dpdqd_3

            ## Step 1: dl/dz1 + dl/dz2*dz2/dz1
            dl_dy_total = dl_dy_3 + dl_dy  # Backward pass vector including result of last step 

            # Use precomputed inverse jacobian
            jac2 = jac[:, keep_constr, :]
            d_int = torch.linalg.solve(jac_pre_inv.transpose(1, 2), # K
                                       dl_dy_total[:, newton_guess_inds].unsqueeze(-1)).squeeze(-1)

            dl_dz_2 = torch.zeros(dl_dy.shape[0], data.npv + data.ng, device=jac.device)
            dl_dz_2[:, data.pg_pv_zidx] = -d_int[:, :data.npv]  # dl_dpg at pv buses
            dl_dz_2[:, data.vm_spv_zidx] = -jac2[:, :, data.vm_start_yidx + data.spv].transpose(1, 2).bmm(
                d_int.unsqueeze(-1)).squeeze(-1)

            dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=jac.device)
            dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
            dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
            dl_dx_2[:, data.nbus + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses

            # Final quantities
            dl_dx_total = dl_dx_3 + dl_dx_2
            dl_dz_total = dl_dz_2 + dl_dy_total[:, np.concatenate([
                data.pg_start_yidx + data.pv_, data.vm_start_yidx + data.spv])]
            # return dl_dx_total, dl_dz_total
            # replace nan in dl_dx_total with 0
            dl_dx_total[torch.isnan(dl_dx_total)] = 0
            dl_dz_total[torch.isnan(dl_dz_total)] = 0
            return dl_dx_total, dl_dz_total
    return PFFunctionFn.apply



class ACOPFProblemV:
    def __init__(self, data, train_num=1000, valid_num=100, test_num=100):
        self.sample = data['sample']
        ppc = data['ppc']
        self.ppc = ppc
        # reset the bus index to start from 0
        self.ppc['bus'][:, idx_bus.BUS_I] -= 1
        self.ppc['gen'][:, idx_gen.GEN_BUS] -= 1
        self.ppc['branch'][:, [0, 1]] -= 1

        self.genbase = ppc['gen'][:, idx_gen.MBASE]
        self.baseMVA = ppc['baseMVA']

        demand = data['Dem'] / self.baseMVA
        gen = data['Gen'] / self.genbase
        self.voltage = data['Vol']

        self.data = data

        self.nbus = self.voltage.shape[1]

        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))

        # indices within gens
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.ng = ppc['gen'].shape[0]
        self.nb = ppc['bus'].shape[0]
        self.nslack = len(self.slack)
        self.npv = len(self.pv)
        self.nspv = len(self.spv)

        self.quad_costs = torch.tensor(ppc['gencost'][:,4], dtype=torch.get_default_dtype())
        self.lin_costs  = torch.tensor(ppc['gencost'][:,5], dtype=torch.get_default_dtype())
        self.const_cost = ppc['gencost'][:,6].sum()

        self.pmax = torch.tensor(ppc['gen'][:,idx_gen.PMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(ppc['gen'][:,idx_gen.PMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(ppc['gen'][:,idx_gen.QMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(ppc['gen'][:,idx_gen.QMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(ppc['bus'][:,idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(ppc['bus'][:,idx_bus.VMIN], dtype=torch.get_default_dtype())
        self.slackva = torch.tensor([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])], 
            dtype=torch.get_default_dtype()).squeeze(-1)
        
        self.va_min = -1.0
        self.va_max = 1.0

        # retrieve the Ybus matrix from the input data
        self.Ybus = data['Ybus']
        self.Ybus = self.Ybus.todense()
        self.Ybusr = torch.tensor(np.real(self.Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(self.Ybus), dtype=torch.get_default_dtype())

        X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)
        Y = np.concatenate([np.real(gen), np.imag(gen), np.abs(self.voltage), np.angle(self.voltage)], axis=1)
        feas_mask =  ~np.isnan(Y).any(axis=1)

        self._X = torch.tensor(X[feas_mask], dtype=torch.get_default_dtype())
        self._Y = torch.tensor(Y[feas_mask], dtype=torch.get_default_dtype())
        self._xdim = X.shape[1]
        self._ydim = Y.shape[1]
        self._num = feas_mask.sum()

        self._neq = 2*self.nbus
        self._nineq = 4*self.ng + 2*self.nbus
        self._nknowns = self.nslack

        self.output_dim = self.nspv + self.npv # predict the voltage mag for all spv buses and voltage angle for all pv buses
        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2*self.ng
        self.va_start_yidx = 2*self.ng + self.nbus


        ## Keep parameters indicating how data was generated
        self.EPS_INTERIOR = data['EPS_INTERIOR'][0][0]
        self.CorrCoeff = data['CorrCoeff'][0][0]
        self.MaxChangeLoad = data['MaxChangeLoad'][0][0]

        ## Define train/valid/test split
        self._train_num = train_num
        self._valid_num = valid_num
        self._test_num = test_num
        assert self.train_num + self.valid_num + self.test_num <= self._num


        # initial values for solver
        self.pg_init = torch.tensor(ppc['gen'][:, idx_gen.PG] / self.genbase)
        self.qg_init = torch.tensor(ppc['gen'][:, idx_gen.QG] / self.genbase)
        self.vm_init = torch.tensor(ppc['bus'][:, idx_bus.VM])
        self.va_init = torch.tensor(np.deg2rad(ppc['bus'][:, idx_bus.VA]))

        # voltage angle at slack buses (known)
        self.slack_va = self.va_init[self.slack]


        # indices of useful quantities in predicted solution
        self.vm_zidx = np.arange(self.nbus)
        self.va_zidx = np.setdiff1d(np.arange(self.nbus), self.slack)

        # useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus

        # useful indices for equality constraints
        # constraints for pv buses, just evaluation
        self.pflow_spv_eqidx = self.pflow_start_eqidx + self.pv
        self.qflow_spv_eqidx = self.qflow_start_eqidx + self.pv

        self.pflow_pq_eqidx = self.pflow_start_eqidx + self.pq
        self.qflow_pq_eqidx = self.qflow_start_eqidx + self.pq

        # the indxes of the remaining equality constraints, need to be treated as inequality constraints
        self.non_spv_eqidx = np.setdiff1d(np.arange(2*self.nbus), np.concatenate([self.pflow_spv_eqidx, self.qflow_spv_eqidx]))

        ### For Pytorch
        self._device = None
        # print(self.eq_resid(self.X[0].unsqueeze(0), self.Y[0].unsqueeze(0)).abs().max())
        # print(self.eq_resid2(self.X[0].unsqueeze(0), self.Y[0].unsqueeze(0)).abs().max())


    # def __str__(self):
    #     return 'ACOPF-{}-{}-{}-{}-{}-{}-{}'.format(
    #         self.nbus,
    #         self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,
    #         self.train_num, self.valid_num, self.test_num)
    
    def __str__(self):
        return 'ACOPF-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            self.nbus,
            self.sample,
            self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,
            self.train_num, self.valid_num, self.test_num)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

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
    def valid_num(self):
        return self._valid_num

    @property
    def test_num(self):
        return self._test_num

    @property
    def train_num(self):
        return self._train_num

    @property
    def trainX(self):
        return self.X[:self.train_num]
    @property
    def validX(self):
        return self.X[self.train_num:self.train_num+self.valid_num]

    @property
    def testX(self):
        return self.X[self.train_num+self.valid_num:self.train_num+self.valid_num+self.test_num]

    @property
    def trainY(self):
        return self.Y[:self.train_num]
    @property
    def validY(self):
        return self.Y[self.train_num:self.train_num+self.valid_num]

    @property
    def testY(self):
        return self.Y[self.train_num+self.valid_num:self.train_num+self.valid_num+self.test_num]

    @property
    def device(self):
        return self._device

    def get_yvars(self, Y):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2*self.ng]
        vm = Y[:, -2*self.nbus:-self.nbus]
        va = Y[:, -self.nbus:]
        return pg, qg, vm, va

    def obj_fn(self, Y):
        pg, _, _, _ = self.get_yvars(Y)
        pg_mw = pg * torch.tensor(self.genbase).to(self.device)
        cost = (self.quad_costs * pg_mw**2).sum(axis=1) + \
            (self.lin_costs * pg_mw).sum(axis=1) + \
            self.const_cost
        return cost / (self.genbase.mean() ** 2)

    def eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        tmp1 = torch.squeeze(torch.matmul(self.Ybusr, vr.unsqueeze(-1)) - torch.matmul(self.Ybusi, vi.unsqueeze(-1)))
        tmp2 = -torch.squeeze(torch.matmul(self.Ybusi, vr.unsqueeze(-1)) + torch.matmul(self.Ybusr, vi.unsqueeze(-1)))

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self.device)
        pg_expand[:, self.spv] = pg 
        real_resid = (pg_expand - X[:, :self.nbus]) - (vr*tmp1 - vi*tmp2)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self.device)
        qg_expand[:, self.spv] = qg 
        react_resid = (qg_expand - X[:, self.nbus:]) - (vr*tmp2 + vi*tmp1)

        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)
        return resids

    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        resids = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            vm - self.vmax,
            self.vmin - vm
        ], dim=1)
        return resids

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        eq_resid = self.eq_resid(X,Y)
        return 2*eq_jac.transpose(1,2).bmm(eq_resid.unsqueeze(-1)).squeeze(-1)

    def eq_jac(self, Y):
        _, _, vm, va = self.get_yvars(Y)

        # helper functions
        mdiag = lambda v1, v2: torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        Ydiagv = lambda Y, v: Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        dtm = lambda v, M: torch.diag_embed(v).bmm(M)

        # helper quantities
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        Yr = self.Ybusr
        Yi = self.Ybusi

        YrvrYivi = torch.squeeze(torch.matmul(self.Ybusr, vr.unsqueeze(-1)) - torch.matmul(self.Ybusi, vi.unsqueeze(-1)))
        YivrYrvi = torch.squeeze(torch.matmul(self.Ybusi, vr.unsqueeze(-1)) + torch.matmul(self.Ybusr, vi.unsqueeze(-1)))

        dreal_dvm = (-mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva)) \
            -mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva)))
        dreal_dva = (-mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr)) \
            -mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr)))
        
        dreact_dvm = (mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva)) \
            -mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva)))
        dreact_dva = (mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr)) \
            -mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr)))
        
        jac = torch.cat([
            torch.cat([dreal_dvm, dreal_dva], dim=2),
            torch.cat([dreact_dvm, dreact_dva], dim=2)],
            dim=1
        )
        return jac

    def scale_partial(self, Z):
        Y_partial = torch.zeros(Z.shape, device=self.device)

        # Y_partial[:, :self.nspv] = Z[:, :self.nspv] * (self.vmax - self.vmin)[:self.nspv] + self.vmin[:self.nspv]
        Y_partial[:, :self.nspv] = Z[:, :self.nspv] * (self.vmax - self.vmin)[self.spv] + self.vmin[self.spv]
        Y_partial[:, self.nspv:] = Z[:, self.nspv:] * (self.va_max - self.va_min) + self.va_min
        return Y_partial

    def complete_partial(self, X, Y_partial):
        # Y_partial = self.scale_partial(Z)
        return PFFunctionV(self)(X, Y_partial)


    def opt_solve(self, X, solver_type='pypower', tol=1e-6):
        X_np = X.detach().cpu().numpy()
        ppc = self.ppc

        # Set reduced voltage bounds if applicable
        ppc['bus'][:,idx_bus.VMIN] = ppc['bus'][:,idx_bus.VMIN] + self.EPS_INTERIOR
        ppc['bus'][:,idx_bus.VMAX] = ppc['bus'][:,idx_bus.VMAX] - self.EPS_INTERIOR

        # Solver options
        ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol)  # MIPS PDIPM

        Y = []
        total_time = 0
        for i in range(X_np.shape[0]):
            print(i)
            ppc['bus'][:, idx_bus.PD] = X_np[i, :self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = X_np[i, self.nbus:] * self.baseMVA

            start_time = time.time()
            my_result = opf(ppc, ppopt)
            # printpf(my_result)
            end_time = time.time()
            total_time += (end_time - start_time)

            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))

        return np.array(Y), total_time, total_time/len(X_np)

def PFFunctionV(data, tol=1e-5, bsz=200, max_iters=15):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            ## Step 1: Newton's method
            Y = torch.zeros(X.shape[0], data.ydim, device=DEVICE)

            Y[:, data.vm_start_yidx + data.spv] = Z[:, :data.nspv] # vm for all spv buses from prediction
            Y[:, data.va_start_yidx + data.pv] = Z[:, data.nspv: ] # va for all pv buses from the prediction (no slack)
            Y[:, data.va_start_yidx + data.slack] = data.slack_va # va at slack bus is known

            # Y[:, data.va_start_yidx + data.slack] = torch.tensor(data.slack_va, device=DEVICE) # va at slack bus is known

            # set the initial values for all vm and va at pq buses
            # Y[:, data.vm_start_yidx + data.pq] = torch.tensor(data.vm_init[data.pq]+0.01, device=DEVICE)
            # Y[:, data.va_start_yidx + data.pq] = torch.tensor(data.va_init[data.pq]+0.001, device=DEVICE)

            Y[:, data.vm_start_yidx + data.pq] = data.vm_init[data.pq]+0.01
            Y[:, data.va_start_yidx + data.pq] = data.va_init[data.pq]+0.001

            newton_guess_inds = np.concatenate([
                data.vm_start_yidx + data.pq,
                data.va_start_yidx + data.pq
            ])

            newton_guess_inds_z = np.concatenate([
                data.pq,
                data.nbus + data.pq
            ])

            keep_constr = np.concatenate([
                data.pflow_start_eqidx + data.pq,     # real power flow at non-slack gens (B-D-R)
                data.qflow_start_eqidx + data.pq
            ])  
        
            # converged = torch.zeros(X.shape[0], dtype=torch.bool, device=DEVICE, requires_grad=False)
            converged = torch.zeros(X.shape[0], device=DEVICE, requires_grad=False)
            jacs = []
            for b in range(0, X.shape[0], bsz):
                X_b = X[b:b+bsz]
                Y_b = Y[b:b+bsz]

                for i in range(max_iters):
                    # print(i)
                    gy = data.eq_resid(X_b, Y_b)[:, keep_constr]
                    jac_full = data.eq_jac(Y_b)
                    jac = jac_full[:, keep_constr, :]  # J_step1
                    # newton_jac_inv = torch.inverse(jac[:, :, newton_guess_inds_z])
                    jac = jac[:, :, newton_guess_inds_z]

                    try:
                        delta = torch.linalg.solve(jac, gy.unsqueeze(-1)).squeeze(-1)
                    except:
                        # if the prediction does not provide a invertible jacobian, take one step and then break
                        # add perturbation to the diagonal of jac
                        perturb = torch.eye(jac.size(1)).unsqueeze(0).repeat(jac.size(0), 1, 1).to(DEVICE)
                        eps = 1e-10
                        delta = torch.linalg.solve(jac + eps*perturb, gy.unsqueeze(-1)).squeeze(-1)
                        Y_b[:, newton_guess_inds] -= delta
                        break
                    Y_b[:, newton_guess_inds] -= delta
                    if torch.norm(delta, dim=1).abs().max() < tol:
                        break
                else:
                    print("the nr completion does not converge!")

                converged[b:b+bsz] = (delta.abs() < tol).all(dim=1)
                if (converged[b:b+bsz] == 0).sum() > 0:
                    print("number of non-converged samples: {}".format((converged[b:b+bsz] == 0).sum()))
                # print("number of unsolvable samples: {}".format((status == 0).sum()))
                jacs.append(jac_full) # only save the last step, TODO: replace by the jacobian evaluated at the current

            # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
            Y[:, data.pg_start_yidx:data.pg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.pflow_start_eqidx + data.spv]
            # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.qflow_start_eqidx + data.spv]
            
            ctx.data = data 
        
            ctx.save_for_backward(torch.cat(jacs), 
            torch.tensor(newton_guess_inds, device=DEVICE), 
            torch.tensor(newton_guess_inds_z, device=DEVICE),
            torch.tensor(keep_constr, device=DEVICE))

            return Y, converged

        @staticmethod
        def backward(ctx, dl_dy, convereged):

            data = ctx.data
            jac, newton_guess_inds, newton_guess_inds_z, keep_constr = ctx.saved_tensors
            ## Step 2 (calc pg at slack and qg at gens)

            # gradient of all voltages through step 3 outputs
            last_eqs = np.concatenate([data.pflow_start_eqidx + data.spv, data.qflow_start_eqidx + data.spv])
            last_vars = np.concatenate([
                np.arange(data.pg_start_yidx, data.pg_start_yidx + data.ng), 
                np.arange(data.qg_start_yidx, data.qg_start_yidx + data.ng)])

            jac3 = jac[:, last_eqs, :]
            dl_dvmva_3 = -jac3.transpose(1,2).bmm(
                dl_dy[:, last_vars].unsqueeze(-1)).squeeze(-1)
            
            dl_dpdqd_3 = dl_dy[:, last_vars]  # dl_dz2

            dl_dy_3 = torch.zeros(dl_dy.shape, device=DEVICE)
            dl_dy_3[:, data.vm_start_yidx:] = dl_dvmva_3

            # gradient of pd at slack and qd at gens through step 3 outputs
            dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)
            dl_dx_3[:, np.concatenate([data.spv, data.nbus + data.spv])] = dl_dpdqd_3

            ## Step 1
            dl_dy_total = dl_dy_3 + dl_dy  # Backward pass vector including result of last step, expand to all output variables

            # Use precomputed inverse jacobian
            jac2 = jac[:, keep_constr, :]

            jac4 = jac[:, keep_constr, :]
            jac4 = jac4[:, :, newton_guess_inds_z]
            try: # K
                d_int = torch.linalg.solve(jac4.transpose(1,2), dl_dy_total[:,newton_guess_inds].unsqueeze(-1)).squeeze(-1)
            except:
                # if the prediction does not provide a invertible jacobian, take one step and then break
                # add perturbation to the diagonal of jac
                perturb = torch.eye(jac4.size(1)).unsqueeze(0).repeat(jac4.size(0), 1, 1).to(DEVICE)
                eps = 1e-10
                d_int = torch.linalg.solve(jac4 + eps*perturb, dl_dy_total[:,newton_guess_inds].unsqueeze(-1)).squeeze(-1)

            dl_dz_2 = torch.zeros(dl_dy.shape[0], data.output_dim, device=DEVICE)
            dl_dz_2[:, :data.ng] = -jac2[:, :, data.spv].transpose(1,2).bmm(
                d_int.unsqueeze(-1)).squeeze(-1)
            dl_dz_2[:, data.ng:] = -jac2[:, :, data.nbus + data.pv].transpose(1,2).bmm(
                d_int.unsqueeze(-1)).squeeze(-1)

            dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)
            dl_dx_2[:, data.pq] = d_int[:, :len(data.pq)]
            dl_dx_2[:, data.nbus + data.pq] = d_int[:, len(data.pq):]

            # Final quantities
            dl_dx_total = dl_dx_3 + dl_dx_2
            dl_dz_total = dl_dz_2 + dl_dy_total[:, np.concatenate([
                data.vm_start_yidx + data.spv, data.va_start_yidx + data.pv])]

            # return dl_dx_total, dl_dz_total
            dl_dx_total[torch.isnan(dl_dx_total)] = 0
            dl_dz_total[torch.isnan(dl_dz_total)] = 0
            return dl_dx_total, dl_dz_total


    return PFFunctionFn.apply


######################################
# ACOPF2
######################################
class ACOPFProblem2:
    """
        minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
        s.t.                  p_g min   <= p_g  <= p_g max
                              q_g min   <= q_g  <= q_g max
                              vmag min  <= vmag <= vmag max
                              vang_slack = \theta_slack   # voltage angle     
                              (p_g - p_d) + (q_g - q_d)i = diag(vmag e^{i*vang}) conj(Y) (vmag e^{-i*vang})
    """

    def __init__(self, data, train_num=1000, valid_num=100, test_num=100):
        self.sample = data['sample']
        ppc = data['ppc']
        self.ppc = ppc
        # reset the bus index to start from 0
        self.ppc['bus'][:, idx_bus.BUS_I] -= 1
        self.ppc['gen'][:, idx_gen.GEN_BUS] -= 1
        self.ppc['branch'][:, [0, 1]] -= 1

        self.genbase = ppc['gen'][:, idx_gen.MBASE]
        self.baseMVA = ppc['baseMVA']

        demand = data['Dem'] / self.baseMVA
        gen = data['Gen'] / self.genbase
        self.voltage = data['Vol']

        self.data = data

        self.nbus = self.voltage.shape[1]

        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))

        # indices within gens
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.ng = ppc['gen'].shape[0]
        self.nb = ppc['bus'].shape[0]
        self.nslack = len(self.slack)
        self.npv = len(self.pv)
        assert self.nb == self.nbus

        self.quad_costs = torch.tensor(ppc['gencost'][:,4], dtype=torch.get_default_dtype())
        self.lin_costs  = torch.tensor(ppc['gencost'][:,5], dtype=torch.get_default_dtype())
        self.const_cost = ppc['gencost'][:,6].sum()

        self.pmax = torch.tensor(ppc['gen'][:,idx_gen.PMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(ppc['gen'][:,idx_gen.PMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(ppc['gen'][:,idx_gen.QMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(ppc['gen'][:,idx_gen.QMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(ppc['bus'][:,idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(ppc['bus'][:,idx_bus.VMIN], dtype=torch.get_default_dtype())
        self.slackva = torch.tensor([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])], 
            dtype=torch.get_default_dtype()).squeeze(-1)

        # retrieve the Ybus matrix from the input data
        self.Ybus = data['Ybus']
        self.Ybus = self.Ybus.todense()
        self.Ybusr = torch.tensor(np.real(self.Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(self.Ybus), dtype=torch.get_default_dtype())

        X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)
        Y = np.concatenate([np.real(gen), np.imag(gen), np.abs(self.voltage), np.angle(self.voltage)], axis=1)
        feas_mask =  ~np.isnan(Y).any(axis=1)

        self._X = torch.tensor(X[feas_mask], dtype=torch.get_default_dtype())
        self._Y = torch.tensor(Y[feas_mask], dtype=torch.get_default_dtype())
        self._xdim = X.shape[1]
        self._ydim = Y.shape[1]
        self._num = feas_mask.sum()

        self._neq = 2*self.nbus
        self._nineq = 4*self.ng + 2*self.nbus
        self._nknowns = self.nslack

        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2*self.ng
        self.va_start_yidx = 2*self.ng + self.nbus


        ## Keep parameters indicating how data was generated
        self.EPS_INTERIOR = data['EPS_INTERIOR'][0][0]
        self.CorrCoeff = data['CorrCoeff'][0][0]
        self.MaxChangeLoad = data['MaxChangeLoad'][0][0]


        ## Define train/valid/test split
        # self._valid_frac = valid_frac
        # self._test_frac = test_frac
        self._train_num = train_num
        self._valid_num = valid_num
        self._test_num = test_num
        assert self.train_num + self.valid_num + self.test_num <= self._num

        ## Define variables and indices for "partial completion" neural network

        # pg (non-slack) and |v|_g (including slack)
        self._partial_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv, self.va_start_yidx + self.slack])
        self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)
        self._partial_unknown_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv])

        # initial values for solver
        self.pg_init = torch.tensor(ppc['gen'][:, idx_gen.PG] / self.genbase)
        self.qg_init = torch.tensor(ppc['gen'][:, idx_gen.QG] / self.genbase)
        self.vm_init = torch.tensor(ppc['bus'][:, idx_bus.VM])
        self.va_init = torch.tensor(np.deg2rad(ppc['bus'][:, idx_bus.VA]))

        # voltage angle at slack buses (known)
        self.slack_va = self.va_init[self.slack]

        # indices of useful quantities in partial solution
        self.pg_pv_zidx = np.arange(self.npv)
        self.vm_spv_zidx = np.arange(self.npv, 2*self.npv + self.nslack)

        # useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus


        ### For Pytorch
        self._device = None
        # print(self.eq_resid(self.X[0].unsqueeze(0), self.Y[0].unsqueeze(0)).abs().max())
        # print(self.eq_resid2(self.X[0].unsqueeze(0), self.Y[0].unsqueeze(0)).abs().max())

        self.Y_init = torch.concatenate([self.pg_init, self.qg_init, self.vm_init, self.va_init]).unsqueeze(0)
        # print(self.eq_resid(self.X[0].unsqueeze(0), Y_init).abs().max())

        self.maxiter = 10


    # def __str__(self):
    #     return 'ACOPF-{}-{}-{}-{}-{}-{}-{}'.format(
    #         self.nbus,
    #         self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,
    #         self.train_num, self.valid_num, self.test_num)
    
    def __str__(self):
        return 'ACOPF-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            self.nbus,
            self.sample,
            self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,
            self.train_num, self.valid_num, self.test_num)

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
        return self._partial_unknown_vars

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
    def valid_num(self):
        return self._valid_num

    @property
    def test_num(self):
        return self._test_num

    @property
    def train_num(self):
        return self._train_num

    @property
    def trainX(self):
        return self.X[:self.train_num]
    @property
    def validX(self):
        return self.X[self.train_num:self.train_num+self.valid_num]

    @property
    def testX(self):
        return self.X[self.train_num+self.valid_num:self.train_num+self.valid_num+self.test_num]

    @property
    def trainY(self):
        return self.Y[:self.train_num]
    @property
    def validY(self):
        return self.Y[self.train_num:self.train_num+self.valid_num]

    @property
    def testY(self):
        return self.Y[self.train_num+self.valid_num:self.train_num+self.valid_num+self.test_num]

    @property
    def device(self):
        return self._device

    def get_yvars(self, Y):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2*self.ng]
        vm = Y[:, -2*self.nbus:-self.nbus]
        va = Y[:, -self.nbus:]
        return pg, qg, vm, va

    def obj_fn(self, Y):
        pg, _, _, _ = self.get_yvars(Y)
        pg_mw = pg * torch.tensor(self.genbase).to(self.device)
        cost = (self.quad_costs * pg_mw**2).sum(axis=1) + \
            (self.lin_costs * pg_mw).sum(axis=1) + \
            self.const_cost
        return cost / (self.genbase.mean() ** 2)

    def eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        tmp1 = torch.squeeze(torch.matmul(self.Ybusr, vr.unsqueeze(-1)) - torch.matmul(self.Ybusi, vi.unsqueeze(-1)))
        tmp2 = -torch.squeeze(torch.matmul(self.Ybusi, vr.unsqueeze(-1)) + torch.matmul(self.Ybusr, vi.unsqueeze(-1)))

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self.device)
        pg_expand[:, self.spv] = pg 
        real_resid = (pg_expand - X[:, :self.nbus]) - (vr*tmp1 - vi*tmp2)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self.device)
        qg_expand[:, self.spv] = qg 
        react_resid = (qg_expand - X[:, self.nbus:]) - (vr*tmp2 + vi*tmp1)

        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)
        return resids

    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        resids = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            vm - self.vmax,
            self.vmin - vm
        ], dim=1)
        return resids

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        eq_resid = self.eq_resid(X,Y)
        return 2*eq_jac.transpose(1,2).bmm(eq_resid.unsqueeze(-1)).squeeze(-1)

    def ineq_grad(self, X, Y):
        ineq_jac = self.ineq_jac(Y)
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_jac.transpose(1,2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)

    def ineq_partial_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.linalg.solve(eq_jac[:, :, self.other_vars], eq_jac[:, :, self.partial_vars])

        direct_grad = self.ineq_grad(X, Y)
        indirect_partial_grad = dynz_dz.transpose(1,2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars]

        full_grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
        full_grad[:, self.partial_vars] = full_partial_grad
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)

        return full_grad

    def eq_jac(self, Y):
        _, _, vm, va = self.get_yvars(Y)

        # helper functions
        mdiag = lambda v1, v2: torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        Ydiagv = lambda Y, v: Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        dtm = lambda v, M: torch.diag_embed(v).bmm(M)

        # helper quantities
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        Yr = self.Ybusr
        Yi = self.Ybusi

        YrvrYivi = torch.squeeze(torch.matmul(self.Ybusr, vr.unsqueeze(-1)) - torch.matmul(self.Ybusi, vi.unsqueeze(-1)))
        YivrYrvi = torch.squeeze(torch.matmul(self.Ybusi, vr.unsqueeze(-1)) + torch.matmul(self.Ybusr, vi.unsqueeze(-1)))

        # real power equations
        dreal_dpg = torch.zeros(self.nbus, self.ng, device=self.device) 
        dreal_dpg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreal_dvm = -mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva)) \
            -mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva))
        dreal_dva = -mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr)) \
            -mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr))
        
        # reactive power equations
        dreact_dqg = torch.zeros(self.nbus, self.ng, device=self.device)
        dreact_dqg[self.spv, :] = torch.eye(self.ng, device=self.device)
        dreact_dvm = mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva)) \
            -mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva))
        dreact_dva = mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr)) \
            -mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr))

        jac = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape), 
                torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreal_dvm, dreal_dva], dim=2),
            torch.cat([torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreact_dqg.unsqueeze(0).expand(vr.shape[0], *dreact_dqg.shape),
                dreact_dvm, dreact_dva], dim=2)],
            dim=1)

        return jac


    def ineq_jac(self, Y):
        jac = torch.cat([
            torch.cat([torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([-torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device),
                torch.eye(self.ng, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=self.device), 
                -torch.eye(self.ng, device=self.device),
                torch.zeros(self.ng, self.nbus, device=self.device), 
                torch.zeros(self.ng, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device),
                torch.zeros(self.nbus, self.ng, device=self.device), 
                torch.eye(self.nbus, device=self.device), 
                torch.zeros(self.nbus, self.nbus, device=self.device)], dim=1),
            torch.cat([torch.zeros(self.nbus, self.ng, device=self.device), 
                torch.zeros(self.nbus, self.ng, device=self.device),
                -torch.eye(self.nbus, device=self.device), 
                torch.zeros(self.nbus, self.nbus, device=self.device)], dim=1)
            ], dim=0)
        return jac.unsqueeze(0).expand(Y.shape[0], *jac.shape)

    # Processes intermediate neural network output
    def process_output(self, X, out):
        out2 = nn.Sigmoid()(out[:, :-self.nbus+self.nslack])
        pg = out2[:, :self.qg_start_yidx] * self.pmax + (1-out2[:, :self.qg_start_yidx]) * self.pmin
        qg = out2[:, self.qg_start_yidx:self.vm_start_yidx] * self.qmax + \
            (1-out2[:, self.qg_start_yidx:self.vm_start_yidx]) * self.qmin
        vm = out2[:, self.vm_start_yidx:] * self.vmax + (1- out2[:, self.vm_start_yidx:]) * self.vmin

        va = torch.zeros(X.shape[0], self.nbus, device=self.device)
        va[:, self.nonslack_idxes] = out[:, self.va_start_yidx:]
        va[:, self.slack] = torch.tensor(self.slack_va, device=self.device).unsqueeze(0).expand(X.shape[0], self.nslack)

        return torch.cat([pg, qg, vm, va], dim=1)
    
    def scale_partial(self, Z):
        Y_partial = torch.zeros(Z.shape, device=self.device)

        Y_partial[:, self.pg_pv_zidx] = Z[:, self.pg_pv_zidx] * (self.pmax[self.pv_] - self.pmin[self.pv_]) + self.pmin[
            self.pv_]
        # Re-scale voltage magnitudes
        Y_partial[:, self.vm_spv_zidx] = Z[:, self.vm_spv_zidx] * (self.vmax[self.spv] - self.vmin[self.spv]) + \
                                        self.vmin[self.spv]
        return Y_partial
    
    def scale_back(self, Y):
        # extract the partial variables from Y
        Y_partial = Y[:, self._partial_vars[:-1]]
        Z = torch.zeros(Y_partial.shape, device=self.device)

        Z[:, self.pg_pv_zidx] = (Y_partial[:, self.pg_pv_zidx] - self.pmin[self.pv_]) / (self.pmax[self.pv_] - self.pmin[self.pv_])
        Z[:, self.vm_spv_zidx] = (Y_partial[:, self.vm_spv_zidx] - self.vmin[self.spv]) / (self.vmax[self.spv] - self.vmin[self.spv])
        # replace nan values with 0
        Z[torch.isnan(Z)] = 0
        return Z

    def complete_partial(self, X, Y_partial):
        return PFFunction2(self, max_iters=self.maxiter)(X, Y_partial)

    def recover_load(self, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        tmp1 = torch.squeeze(torch.matmul(self.Ybusr, vr.unsqueeze(-1)) - torch.matmul(self.Ybusi, vi.unsqueeze(-1)))
        tmp2 = -torch.squeeze(torch.matmul(self.Ybusi, vr.unsqueeze(-1)) + torch.matmul(self.Ybusr, vi.unsqueeze(-1)))

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self.device)
        pg_expand[:, self.spv] = pg 
        pd = pg_expand - vr*tmp1 + vi*tmp2

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self.device)
        qg_expand[:, self.spv] = qg 
        qd = qg_expand - vr*tmp2 - vi*tmp1

        return torch.cat([pd, qd], dim=1)

    def opt_solve(self, X, solver_type='pypower', tol=1e-6):
        X_np = X.detach().cpu().numpy()
        ppc = self.ppc

        # Set reduced voltage bounds if applicable
        ppc['bus'][:,idx_bus.VMIN] = ppc['bus'][:,idx_bus.VMIN] + self.EPS_INTERIOR
        ppc['bus'][:,idx_bus.VMAX] = ppc['bus'][:,idx_bus.VMAX] - self.EPS_INTERIOR

        # Solver options
        ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol)  # MIPS PDIPM

        Y = []
        total_time = 0
        for i in range(X_np.shape[0]):
            print(i)
            ppc['bus'][:, idx_bus.PD] = X_np[i, :self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = X_np[i, self.nbus:] * self.baseMVA

            start_time = time.time()
            my_result = opf(ppc, ppopt)
            # printpf(my_result)
            end_time = time.time()
            total_time += (end_time - start_time)

            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))

        return np.array(Y), total_time, total_time/len(X_np)

def PFFunction2(data, tol=1e-5, bsz=200, max_iters=20):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            # start_time = time.time()
            ## Step 1: Newton's method
            Y = torch.zeros(X.shape[0], data.ydim, device=X.device)
            # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
            Y[:, data.pg_start_yidx + data.pv_] = Z[:, data.pg_pv_zidx]  # pg at non-slack gens
            Y[:, data.vm_start_yidx + data.spv] = Z[:, data.vm_spv_zidx]  # vm at gens
            # init guesses for remaining values
            Y[:, data.vm_start_yidx + data.pq] = data.vm_init[data.pq]  # vm at load buses
            Y[:, data.va_start_yidx: data.va_start_yidx+data.nb] = data.va_init   # va at all bus
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = 0  # qg at gens (not used in Newton upd)
            Y[:, data.pg_start_yidx + data.slack_] = 0  # pg at slack (not used in Newton upd)

            keep_constr = np.concatenate([
                data.pflow_start_eqidx + data.pv,  # real power flow at non-slack gens
                data.pflow_start_eqidx + data.pq,  # real power flow at load buses
                data.qflow_start_eqidx + data.pq])  # reactive power flow at load buses
            newton_guess_inds = np.concatenate([
                data.vm_start_yidx + data.pq,  # vm at load buses
                data.va_start_yidx + data.pv,  # va at non-slack gens
                data.va_start_yidx + data.pq])  # va at load buses

            converged = torch.zeros(X.shape[0])
            jacs = []
            # newton_jacs_inv = []
            for b in range(0, X.shape[0], bsz):
                X_b = X[b:b + bsz]
                Y_b = Y[b:b + bsz]
                for _ in range(max_iters):
                    gy = data.eq_resid(X_b, Y_b)[:, keep_constr]
                    jac_full = data.eq_jac(Y_b)
                    jac = jac_full[:, keep_constr, :]
                    jac = jac[:, :, newton_guess_inds]

                    """Linear system"""
                    try:
                        delta = torch.linalg.solve(jac, gy.unsqueeze(-1)).squeeze(-1)
                    except:
                        # add small value to diagonal to ensure invertibility
                        jac += 1e-5*torch.eye(jac.shape[1], device=jac.device).unsqueeze(0).expand(jac.shape[0], *jac.shape[1:])
                        delta = torch.linalg.solve(jac, gy.unsqueeze(-1)).squeeze(-1)
                        break

                    Y_b[:, newton_guess_inds] -= delta
                    if torch.abs(delta).max() < tol:
                        break
                # if torch.abs(delta).max() > tol:
                #     print('Newton methods for Power Flow does not converge')
                # print(torch.abs(delta).max())
                # converged[b:b + bsz] = (delta.abs() < tol).all(dim=1)
                jacs.append(jac_full)
                converged[b:b+bsz] = (delta.abs() < tol).all(dim=1)
                if (converged[b:b+bsz] == 0).sum() > 0:
                    print("number of non-converged samples: {}".format((converged[b:b+bsz] == 0).sum()))
                # newton_jacs_inv.append(newton_jac_inv)

            ## Step 2: Solve for remaining variables
            # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.qflow_start_eqidx + data.spv]
            # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
            Y[:, data.pg_start_yidx + data.slack_] = \
                -data.eq_resid(X, Y)[:, data.pflow_start_eqidx + data.slack]

            ctx.data = data
            ctx.save_for_backward(torch.cat(jacs),
                                  torch.as_tensor(newton_guess_inds, device=X.device),
                                  torch.as_tensor(keep_constr, device=X.device))
            return Y, converged

        @staticmethod
        def backward(ctx, dl_dy, dl_dc):

            data = ctx.data
            # jac, newton_jac_inv, newton_guess_inds, keep_constr = ctx.saved_tensors
            jac, newton_guess_inds, keep_constr = ctx.saved_tensors

            ## Step 2 (calc pg at slack and qg at gens)
            jac_pre_inv = jac[:, keep_constr, :]
            jac_pre_inv = jac_pre_inv[:, :, newton_guess_inds]

            # gradient of all voltages through step 3 outputs
            last_eqs = np.concatenate([data.pflow_start_eqidx + data.slack, data.qflow_start_eqidx + data.spv])
            last_vars = np.concatenate([
                data.pg_start_yidx + data.slack_, np.arange(data.qg_start_yidx, data.qg_start_yidx + data.ng)])
            # last_vars = np.concatenate([
            #     data.pg_start_yidx + data.slack_, data.qg_start_yidx + data.spv])
            jac3 = jac[:, last_eqs, :]
            dl_dvmva_3 = -jac3[:, :, data.vm_start_yidx:].transpose(1, 2).bmm( # dl/dz2*dz2/dz1
                dl_dy[:, last_vars].unsqueeze(-1)).squeeze(-1)

            # gradient of pd at slack and qd at gens through step 3 outputs
            dl_dpdqd_3 = dl_dy[:, last_vars] 

            # insert into correct places in x and y loss vectors
            dl_dy_3 = torch.zeros(dl_dy.shape, device=jac.device)
            dl_dy_3[:, data.vm_start_yidx:] = dl_dvmva_3

            dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=jac.device)
            dl_dx_3[:, np.concatenate([data.slack, data.nbus + data.spv])] = dl_dpdqd_3

            ## Step 1: dl/dz1 + dl/dz2*dz2/dz1
            dl_dy_total = dl_dy_3 + dl_dy  # Backward pass vector including result of last step 

            # Use precomputed inverse jacobian
            jac2 = jac[:, keep_constr, :]
            try:
                d_int = torch.linalg.solve(jac_pre_inv.transpose(1, 2), # K
                                       dl_dy_total[:, newton_guess_inds].unsqueeze(-1)).squeeze(-1)
            except:
                # if the prediction does not provide a invertible jacobian, take one step and then break
                # add perturbation to the diagonal of jac
                perturb = torch.eye(jac_pre_inv.size(1)).unsqueeze(0).repeat(jac_pre_inv.size(0), 1, 1).to(DEVICE)
                eps = 1e-10
                d_int = torch.linalg.solve(jac_pre_inv + eps*perturb, dl_dy_total[:, newton_guess_inds].unsqueeze(-1)).squeeze(-1)
            dl_dz_2 = torch.zeros(dl_dy.shape[0], data.npv + data.ng, device=jac.device)
            dl_dz_2[:, data.pg_pv_zidx] = -d_int[:, :data.npv]  # dl_dpg at pv buses
            dl_dz_2[:, data.vm_spv_zidx] = -jac2[:, :, data.vm_start_yidx + data.spv].transpose(1, 2).bmm(
                d_int.unsqueeze(-1)).squeeze(-1)

            dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=jac.device)
            dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
            dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
            dl_dx_2[:, data.nbus + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses

            # Final quantities
            dl_dx_total = dl_dx_3 + dl_dx_2
            dl_dz_total = dl_dz_2 + dl_dy_total[:, np.concatenate([
                data.pg_start_yidx + data.pv_, data.vm_start_yidx + data.spv])]
            # return dl_dx_total, dl_dz_total
            # replace nan in dl_dx_total with 0
            dl_dx_total[torch.isnan(dl_dx_total)] = 0
            dl_dz_total[torch.isnan(dl_dz_total)] = 0
            return dl_dx_total, dl_dz_total
    return PFFunctionFn.apply