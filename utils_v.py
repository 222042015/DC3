import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)

import numpy as np
import osqp
from qpth.qp import QPFunction
# import ipopt
import cyipopt
from scipy.linalg import svd
from scipy.sparse import csr_matrix as sparse

import hashlib
from copy import deepcopy
import scipy.io as spio
import time
import numba

from pypower.api import case118, case57, case300, case39, case4gs, case30, case24_ieee_rts, case14
from pypower.api import opf, makeYbus, loadcase
from pypower import idx_bus, idx_gen, ppoption, idx_brch

import pandapower as pp

from numpy import zeros, c_, shape, ix_

from pypower.idx_bus import MU_VMIN
from pypower.idx_gen import PG, QG
from pypower.idx_brch import PF, QF, PT, QT, F_BUS, T_BUS, ANGMAX, ANGMIN, RATE_A, RATE_B, RATE_C

from pypower.ext2int import ext2int
from pypower.opf_setup import opf_setup
from pypower.opf_execute import opf_execute
from pypower.int2ext import int2ext

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")


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
CASE_FNS = dict([(300, case300), (57, case57), (118, case118), (39, case39), (4, case4gs), (14, case14), (30, case30), (24, case24_ieee_rts)])

class ACOPFProblem:
    """
        minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
        s.t.                  p_g min   <= p_g  <= p_g max
                              q_g min   <= q_g  <= q_g max
                              vmag min  <= vmag <= vmag max
                              vang_slack = \theta_slack   # voltage angle     
                              (p_g - p_d) + (q_g - q_d)i = diag(vmag e^{i*vang}) conj(Y) (vmag e^{-i*vang})
    """

    def __init__(self, filename, valid_frac=0.0833, test_frac=0.0833):
        data = spio.loadmat(filename)
        self.nbus = int(filename.split('_')[-1][4:-4])
        print("number of bus: {}".format(self.nbus))

        ## Define useful power network quantities and indices
        ppc = CASE_FNS[self.nbus]()
        # ppc = loadcase('/home/jxxiong/A-xjx/DC3/datasets/acopf/case'+str(self.nbus)+'.mat')

        ## change to internal indexing
        ppc = ext2int(ppc)
        self.ppc = ppc

        self.genbase = ppc['gen'][:, idx_gen.MBASE]
        self.baseMVA = ppc['baseMVA']

        # each bus only can has one generator, (number of buses == number of slack bus and pv buses)
        # so the the we only need to record the index of the spv buses as the generators
        # the generator index is naturally determined by the natural index starting from 0
        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0] 
        self.spv = np.concatenate([self.slack, self.pv]) 
        self.spv.sort()  # the indices of the slack and pv buses in ascending order 
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))

        # indices within spv (the indices of the slack and pv buses in self.spv)
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.ng = ppc['gen'].shape[0]
        self.nslack = len(self.slack)
        self.npv = len(self.pv)
        self.nspv = len(self.spv)

        self.quad_costs = torch.tensor(ppc['gencost'][:,4], dtype=torch.get_default_dtype())
        self.lin_costs  = torch.tensor(ppc['gencost'][:,5], dtype=torch.get_default_dtype())
        self.const_cost = ppc['gencost'][:,6].sum()

        # branch data
        self.branch = ppc['branch']
        self.fbus = self.branch[:, F_BUS].astype(int) - 1
        self.tbus = self.branch[:, T_BUS].astype(int) - 1

        self.angmax = torch.tensor(np.deg2rad(self.branch[:, ANGMAX]), dtype=torch.get_default_dtype())
        self.angmin = torch.tensor(np.deg2rad(self.branch[:, ANGMIN]), dtype=torch.get_default_dtype())
        self.pmax = torch.tensor(ppc['gen'][:,idx_gen.PMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(ppc['gen'][:,idx_gen.PMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(ppc['gen'][:,idx_gen.QMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(ppc['gen'][:,idx_gen.QMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(ppc['bus'][:,idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(ppc['bus'][:,idx_bus.VMIN], dtype=torch.get_default_dtype())
        self.slackva = torch.tensor(np.array([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])]), 
            dtype=torch.get_default_dtype()).squeeze(-1)
        self.bfmax = torch.tensor(self.branch[:, RATE_A] / self.baseMVA, dtype=torch.get_default_dtype())
        
        print("the range of the voltage magnitudes: {}, {}".format((self.vmax - self.vmin).min(), (self.vmax - self.vmin).max()))
        print("the range of the active power generation: {}, {}".format((self.pmax - self.pmin).min(), (self.pmax - self.pmin).max()))
        print("the range of the angle difference: {}, {}".format((self.angmax - self.angmin).min(), (self.angmax - self.angmin).max()))
        

        # ppc2 = ext2int(ppc)
        # Ybus, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus, Yf, Yt = makeYbus(self.baseMVA, self.ppc['bus'], self.ppc['branch'])
        Ybus = Ybus.todense()
        Yf = Yf.todense()
        Yt = Yt.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.get_default_dtype())
        self.Yf = torch.tensor(Yf, dtype=torch.complex128)
        self.Yt = torch.tensor(Yt, dtype=torch.complex128)

        ## Define optimization problem input and output variables
        demand = data['Dem'].T / self.baseMVA
        gen =  data['Gen'].T / self.genbase
        voltage = data['Vol'].T

        X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)
        Y = np.concatenate([np.real(gen), np.imag(gen), np.abs(voltage), np.angle(voltage)], axis=1)
        feas_mask =  ~np.isnan(Y).any(axis=1)

        self._X = torch.tensor(X[feas_mask], dtype=torch.get_default_dtype())
        self._Y = torch.tensor(Y[feas_mask], dtype=torch.get_default_dtype())
        self._xdim = X.shape[1]
        self._ydim = Y.shape[1]
        self._num = feas_mask.sum()

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
        self._valid_frac = valid_frac
        self._test_frac = test_frac

        # initial values for solver
        self.vm_init = ppc['bus'][:, idx_bus.VM]
        self.va_init = np.deg2rad(ppc['bus'][:, idx_bus.VA])
        self.pg_init = ppc['gen'][:, idx_gen.PG] / self.genbase
        self.qg_init = ppc['gen'][:, idx_gen.QG] / self.genbase
        print("the range of the voltage angle: {}, {}".format(self.va_init.min(), self.va_init.max()))
        self.va_min = self.va_init.min()
        self.va_max = self.va_init.max()

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

        self._neq = 2*self.nbus
        self._nineq = 4*self.ng + 2*len(self.branch)
        self._nknowns = self.nslack
        self.output_dim = 2*self.nspv - self.nslack  # vm for all pv and slack buses, the va for the slack bus is known
        ### For Pytorch
        self._device = None
        self.bound = True

        # self.pred_demand(self.X, self.Y)
        Sf, St = self.get_branch_flow(self.X, self.Y)



    def __str__(self):
        return 'ACOPF-{}-{}-{}-{}-{}-{}'.format(
            self.nbus,
            self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,
            self.valid_frac, self.test_frac)

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
        return self.X[:int(self.num * self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num * self.train_frac):int(self.num * (self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num * (self.train_frac + self.valid_frac)):]

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
    def device(self):
        return self._device

    def get_yvars(self, Y):
        pg = Y[:, :self.ng].clone()
        qg = Y[:, self.ng:2*self.ng].clone()
        vm = Y[:, -2*self.nbus:-self.nbus].clone()
        va = Y[:, -self.nbus:].clone()
        return pg, qg, vm, va

    def get_zvars(self, Z):
        vm = Z[:, :self.nbus].clone()
        va = torch.zeros(Z.shape[0], self.nbus, device=self.device)
        va[:, self.nonslack_idxes] = Z[:, self.nbus:].clone()
        va[:, self.slack] = torch.tensor(self.slack_va, device=self.device)
        return vm, va
    
    def get_branch_ang(self, va):
        return va[:, self.fbus] - va[:, self.tbus]
    
    def get_branch_flow(self, X, Y):
        _, _, vm, va = self.get_yvars(Y)
        V = vm * torch.exp(1j*va)
        Vf = vm[:, self.fbus]
        Vt = vm[:, self.tbus]
        If = (self.Yf.unsqueeze(0).expand(V.shape[0], *self.Yf.shape).bmm(V.unsqueeze(-1)).squeeze(-1)).conj()
        It = (self.Yt.unsqueeze(0).expand(V.shape[0], *self.Yt.shape).bmm(V.unsqueeze(-1)).squeeze(-1)).conj()
        Sf = Vf * If
        St = Vt * It

        return Sf, St



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

        ## power balance equations
        tmp1 = vr@self.Ybusr - vi@self.Ybusi
        tmp2 = -vr@self.Ybusi - vi@self.Ybusr

        # real power
        # expand the generator real power to all buses, augemented with zeros if the bus is not a pv bus
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
        YrvrYivi = vr@Yr - vi@Yi
        YivrYrvi = vr@Yi + vi@Yr

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
    
    def eq_resid_part(self, X, Y, indx):
        eq_res = self.eq_resid(X, Y)
        return eq_res[:, indx]

    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        angdiff = self.get_branch_ang(va)
        resids = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            angdiff - self.angmax,
            self.angmin - angdiff,
        ], dim=1)
        return resids

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)
    
    def eq_dist(self, X, Y):
        resids = self.eq_resid(X, Y)
        return torch.abs(resids)
    
    def complete_partial(self, X, Z):
        Y_partial = torch.zeros(Z.shape, device=self.device)

        Y_partial[:, :self.nspv] = Z[:, :self.nspv] * (self.vmax - self.vmin)[:self.nspv] + self.vmin[:self.nspv]
        Y_partial[:, self.nspv:] = Z[:, self.nspv:] * (self.va_max - self.va_min) + self.va_min

        return PFFunction(self)(X, Y_partial)


    def pred_demand(self, X, Z):
        '''
        given current prediction Y,
        - for all spv (with generator), derive the corresponding active and reactive generation
        - for all the buses, derive the predicted active and reactive demand
        - the correctness of the function can be verified by evaluating at the ground truth data.
        '''
        # _, _, vm, va = self.get_yvars(Y)
        if Z.shape[1] == self.ydim:
            _, _, vm, va = self.get_yvars(Z)
        else:
            vm, va = self.get_zvars(Z)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        ## power balance equations
        tmp1 = vr@self.Ybusr - vi@self.Ybusi
        tmp2 = -vr@self.Ybusi - vi@self.Ybusr
        
        P = vr*tmp1 - vi*tmp2
        Q = vr*tmp2 + vi*tmp1

        pg_expand = X[:, :self.nbus] + P
        qg_expand = X[:, self.nbus:] + Q

        Pg = pg_expand[:, self.spv]
        Qg = qg_expand[:, self.spv]

        Pd = -P * 1.0
        Qd = -Q * 1.0

        Pd[:, self.spv] = Pg - P[:, self.spv]
        Qd[:, self.spv] = Qg - Q[:, self.spv]

        # check the correctness of the function
        # # calculate the difference between the predicted demand and the ground truth demand
        # Pd_diff = Pd - X[:, :self.nbus]
        # Qd_diff = Qd - X[:, self.nbus:]

        # # the difference between the active and reactive generation
        # Pg_diff = Pg - Z[:, :self.ng]
        # Qg_diff = Qg - Z[:, self.ng:self.ng*2]

        return Pg, Qg, Pd, Qd
    
    

def PFFunction(data, tol=1e-5, bsz=200, max_iters=20):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            ## Step 1: Newton's method
            Y = torch.zeros(X.shape[0], data.ydim, device=DEVICE)

            Y[:, data.vm_start_yidx + data.spv] = Z[:, :data.nspv] # vm for all spv buses from prediction
            Y[:, data.va_start_yidx + data.pv] = Z[:, data.nspv: ] # va for all pv buses from the prediction (no slack)
            Y[:, data.va_start_yidx + data.slack] = torch.tensor(data.slack_va, device=DEVICE) # va at slack bus is known

            # set the initial values for all vm and va at pq buses
            Y[:, data.vm_start_yidx + data.pq] = torch.tensor(data.vm_init[data.pq], device=DEVICE)
            Y[:, data.va_start_yidx + data.pq] = torch.tensor(data.va_init[data.pq] + 0.01, device=DEVICE)

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
        
            converged = torch.zeros(X.shape[0], dtype=torch.bool, device=DEVICE)
            jacs = []
            newton_jacs_inv = []
            for b in range(0, X.shape[0], bsz):
                # print('batch: {}'.format(b))
                X_b = X[b:b+bsz]
                Y_b = Y[b:b+bsz]

                # status = data.check_solvable(X_b, Y_b, False)

                for i in range(max_iters):
                    # print(i)
                    gy = data.eq_resid(X_b, Y_b)[:, keep_constr]
                    jac_full = data.eq_jac(Y_b)
                    jac = jac_full[:, keep_constr, :]  # J_step1
                    # newton_jac_inv = torch.inverse(jac[:, :, newton_guess_inds_z])

                    jac = jac[:, :, newton_guess_inds_z]
                    try:
                        newton_jac_inv = torch.inverse(jac)
                    except:
                        perturb = torch.eye(jac.size(1)).unsqueeze(0).repeat(jac.size(0), 1, 1).to(DEVICE)
                        eps = 1e-16
                        newton_jac_inv = torch.inverse(jac + eps*perturb)

                    delta = newton_jac_inv.bmm(gy.unsqueeze(-1)).squeeze(-1)
                    Y_b[:, newton_guess_inds] -= delta
                    if torch.norm(delta, dim=1).abs().max() < tol:
                        break
                else:
                    print("the nr completion does not converge!")

                converged[b:b+bsz] = (delta.abs() < tol).all(dim=1)
                print("number of non-converged samples: {}".format((converged[b:b+bsz] == 0).sum()))
                # print("number of unsolvable samples: {}".format((status == 0).sum()))
                jacs.append(jac_full) # only save the last step
                newton_jacs_inv.append(newton_jac_inv)


            ## Step 2: Solve for remaining variables
            # Pg, Qg, _, _ = data.pred_demand(X, Y)
            # Y[:, data.pg_start_yidx: data.pg_start_yidx+data.ng] = Pg
            # Y[:, data.qg_start_yidx: data.qg_start_yidx+data.ng] = Qg
            # # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
            Y[:, data.pg_start_yidx:data.pg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.pflow_start_eqidx + data.spv]
            # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.qflow_start_eqidx + data.spv]
            
            if data.bound:
                print("the range of the voltage angle: {}, {}".format(data.va_min, data.va_max))
                if torch.sum(converged) / X.shape[0] > 0.8:
                    # with a probability, the range of the voltage angle is multiplied by 1.2
                    if np.random.rand() > 0.9:
                        data.va_min = max(-np.pi, data.va_min*1.2)
                        data.va_max = min(np.pi, data.va_max*1.2)
                    else:
                        data.va_min = max(data.va_min, Y[converged][:, data.va_start_yidx + data.nonslack_idxes].min())
                        data.va_max = min(data.va_max, Y[converged][:, data.va_start_yidx + data.nonslack_idxes].max())
                else:
                    # thrink the bound for the voltage angle
                    data.va_min = data.va_min / 1.2
                    data.va_max = data.va_max / 1.2
                data.va_min = max(data.va_min, -np.pi)
                data.va_max = min(data.va_max, np.pi)
            

            ctx.data = data 
            ctx.save_for_backward(torch.cat(jacs), torch.cat(newton_jacs_inv),
                torch.tensor(newton_guess_inds, device=DEVICE), 
                torch.tensor(keep_constr, device=DEVICE))

            return Y

        @staticmethod
        def backward(ctx, dl_dy):

            data = ctx.data
            jac, newton_jac_inv, newton_guess_inds, keep_constr = ctx.saved_tensors

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
            d_int = newton_jac_inv.transpose(1,2).bmm(
                            dl_dy_total[:,newton_guess_inds].unsqueeze(-1)).squeeze(-1)  # K

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

            return dl_dx_total, dl_dz_total


    return PFFunctionFn.apply