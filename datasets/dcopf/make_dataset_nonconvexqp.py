import osqp
import time
import random
import pickle
import numpy as np
import gurobipy as gp

from gurobipy import Model, GRB, QuadExpr, LinExpr
from scipy.sparse import csc_matrix
import sys
import os
import scipy.io as spio

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))

from dcopf_utils import build_gurobi_model


filepath = "/home/jxxiong/A-xjx/DC3/datasets/dcopf/qp50_25_4_4.mat"

data = spio.loadmat(filepath)
b = data['beq'].squeeze()
# ranfomly generate X as numsamples of random perturbation of b by factors randomly generated from [0.9, 1.1]
num_samples = 1200
Q = data['H']
p = data['f'].squeeze()
A = data['Aeq']
G = data['A']
h = data['b'].squeeze()
Lb = data['LB'].squeeze()
Ub = data['UB'].squeeze()

X = []
while len(X) < num_samples:
    x = np.random.rand(b.shape[0]) * 0.6 + 0.7
    x = x * b
    full_model = build_gurobi_model(Q, p, A, x, G, h, Lb, Ub)
    full_model.setParam('OutputFlag', 1)
    full_model.setParam('TimeLimit', 10)
    # full_model.setParam('Presolve', -1)
    full_model.optimize()
    if full_model.status == GRB.OPTIMAL and full_model.Runtime > 1:
        X.append(x)
    print(f"X: {len(X)}")
X = np.array(X)
print(X.shape)

dataset = {'Q':data['H'], 'p':data['f'].squeeze(), 'A':data['Aeq'], 'X':X, 'G':data['A'], 'h':data['b'].squeeze(), 'Lb':data['LB'].squeeze(), 'Ub':data['UB'].squeeze(), 'Y':[]}
with open(f"dcopf51_data", 'wb') as f:
    pickle.dump(dataset, f)
        