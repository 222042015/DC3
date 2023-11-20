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
from pypower.idx_brch import PF, QF, PT, QT, F_BUS, T_BUS, ANGMAX, ANGMIN

from pypower.ext2int import ext2int
from pypower.opf_setup import opf_setup
from pypower.opf_execute import opf_execute
from pypower.int2ext import int2ext

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

        ## Define useful power network quantities and indices
        ppc = CASE_FNS[self.nbus]()
        # ppc = loadcase('/home/jxxiong/A-xjx/DC3/datasets/acopf/case'+str(self.nbus)+'.mat')

        ## change to internal indexing
        # ppc = ext2int(ppc)
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
        self.slackva = torch.tensor([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])], 
            dtype=torch.get_default_dtype()).squeeze(-1)
        
        print("the range of the voltage magnitudes: {}, {}".format((self.vmax - self.vmin).min(), (self.vmax - self.vmin).max()))
        print("the range of the active power generation: {}, {}".format((self.pmax - self.pmin).min(), (self.pmax - self.pmin).max()))

        ppc2 = ext2int(ppc)
        Ybus, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.get_default_dtype())

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

        ## Define variables and indices for "partial completion" neural network

        # pg (non-slack) and |v|_g (including slack)
        # self._partial_vars = np.concatenate([self.vm_start_yidx + np.arange(self.nbus), self.va_start_yidx + self.nonslack_idxes])
        # self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)

        # initial values for solver
        self.vm_init = ppc['bus'][:, idx_bus.VM]
        self.va_init = np.deg2rad(ppc['bus'][:, idx_bus.VA])
        self.pg_init = ppc['gen'][:, idx_gen.PG] / self.genbase
        self.qg_init = ppc['gen'][:, idx_gen.QG] / self.genbase

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

        # the indxes of the remaining equality constraints, need to be treated as inequality constraints
        self.non_spv_eqidx = np.setdiff1d(np.arange(2*self.nbus), np.concatenate([self.pflow_spv_eqidx, self.qflow_spv_eqidx]))

        self._neq = 2*self.nbus
        # self._nineq = 4*self.ng + 4*self.nbus + 2*len(self.branch)
        self._nineq = 4*self.ng + 2*len(self.branch)
        self._nknowns = self.nslack
        self.output_dim = 2*self.nbus - self.nslack  # vm for all buses and va for all non-slack buses
        ### For Pytorch
        self._device = None


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
    
    def eq_resid_part(self, X, Y, indx):
        eq_res = self.eq_resid(X, Y)
        return eq_res[:, indx]

    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        angdiff = self.get_branch_ang(va)
        eq_resid = self.eq_resid(X, Y)
        resids = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            angdiff - self.angmax,
            self.angmin - angdiff,
            # eq_resid,
            # -eq_resid
        ], dim=1)
        return resids

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)
    
    def eq_dist(self, X, Y):
        resids = self.eq_resid(X, Y)
        return torch.abs(resids)

    def complete_partial(self, X, Z):
        Z[:, self.nbus:] = Z[:, self.nbus:] * 2*np.pi - np.pi
        Z[:, :self.nbus] = Z[:, :self.nbus] * (self.vmax - self.vmin) + self.vmin

        vm, va = self.get_zvars(Z)
        Pg, Qg, Pd, Qd = self.pred_demand(X, Z)

        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.pg_start_yidx: self.pg_start_yidx+self.ng] = Pg
        Y[:, self.qg_start_yidx: self.qg_start_yidx+self.ng] = Qg
        Y[:, self.vm_start_yidx: self.vm_start_yidx+self.nbus] = vm
        Y[:, self.va_start_yidx: self.va_start_yidx+self.nbus] = va

        return Y, Pd, Qd


    def pred_demand(self, X, Z):
        '''
        given current prediction Y,
        - for all spv (with generator), derive the corresponding active and reactive generation
        - for all the buses, derive the predicted active and reactive demand
        - the correctness of the function can be verified by evaluating at the ground truth data.
        '''
        # _, _, vm, va = self.get_yvars(Y)
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

        return Pg, Qg, Pd, Qd