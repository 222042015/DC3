# generate the samples for the ACOPF problem
import os
import pickle
import pypower.api as pp
from pypower import idx_bus, idx_gen, ppoption, idx_brch
from pypower.api import opf, makeYbus
from pypower.idx_bus import MU_VMIN
from pypower.idx_gen import PG, QG, MU_QMIN, MU_PMAX, MU_PMIN
from pypower.idx_brch import PF, QF, PT, QT, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX
import numpy as np
import scipy.io as sio
import scipy
import sys
from minimax_tilting_sampler import TruncatedMVN

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))

CASE_FNS = dict([(300, pp.case300), (57, pp.case57), (118, pp.case118), (39, pp.case39)])
nbus = 300
nsamples = 1200
randomType = 'uniform'

EPS_INTERIOR = 0


# Load the case
pcc = CASE_FNS[nbus]()
CorrCoeff = 0.3
MaxChangeLoad = 0.1

# Solver options
ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=1e-5)  # MIPS PDIPM

Dem = []
Gen = []
Vol = []

bus = pcc['bus']
br = pcc['branch']
gen = pcc['gen']
MVAbase = pcc['baseMVA']

baseLoadP = bus[:, idx_bus.PD]
baseLoadQ = bus[:, idx_bus.QD]

# select the buses with positive baseLoadP
loadBuses = np.where(baseLoadP > 0)[0]
nloads = len(loadBuses)


n = 0
while n < nsamples:
    # Generate the load
    print(n)
    if randomType == 'uniform':
        loadFactor = np.random.uniform(0.8, 1.2, nloads)
    elif randomType == 'normal':
        # sample from truncated multivariate Gaussian distribution with µ to 0.3 and the
        # covariance matrix Σ to the correlated covariance with the correlation parameter of 0.8

        # Generate the correlated covariance matrix
        scalesSigma = MaxChangeLoad / 1.645
        Sigma = scalesSigma ** 2 * (CorrCoeff * np.ones((nloads, nloads)) + (1 - CorrCoeff) * np.eye(nloads))
        mu = np.ones(nloads) * 1.0
        lb = (1 - MaxChangeLoad) * np.ones(nloads)
        ub = (1 + MaxChangeLoad) * np.ones(nloads)
        
        loadFactor = TruncatedMVN(mu, Sigma, lb, ub).sample(1).reshape(-1)

    else:
        raise NotImplementedError

    pcc['bus'][loadBuses, idx_bus.PD] = baseLoadP[loadBuses] * loadFactor
    pcc['bus'][loadBuses, idx_bus.QD] = baseLoadQ[loadBuses] * loadFactor

    # Solve the ACOPF
    my_result = opf(pcc, ppopt)

    if my_result['success']:
        n += 1
        Dem.append(pcc['bus'][:, idx_bus.PD] + 1j * pcc['bus'][:, idx_bus.QD])
        Gen.append(my_result['gen'][:, idx_gen.PG] + 1j * my_result['gen'][:, idx_gen.QG])
        vm = my_result['bus'][:, idx_bus.VM]
        va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
        Vol.append(vm * np.exp(1j * va))
    
    if n % 100 == 0:
        print(n)
        sio.savemat('./FeasiblePairs_Case{}.mat'.format(nbus), {'Dem': np.array(Dem).T, 'Gen': np.array(Gen).T, 'Vol': np.array(Vol).T, 'EPS_INTERIOR': EPS_INTERIOR, 'CorrCoeff': CorrCoeff, 'MaxChangeLoad': MaxChangeLoad})

