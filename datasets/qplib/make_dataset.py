# %%
'''
generate dataset from qplib
from Xi Gao
'''

import osqp
import time
import tqdm
import random
import pickle
import numpy as np
import gurobipy as gp

from gurobipy import Model, GRB, QuadExpr, LinExpr
from scipy.sparse import csc_matrix
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from dcopf_utils import DcopfProblem

# %% [markdown]
# ## Import Data

# %%
# qp_path = ''
nbus = 9002
m = gp.read(f'/home/jxxiong/A-xjx/DC3/datasets/qplib/QPLIB_{nbus}.lp')
n_samples = 10000
perturb_rate = 0.3

# create a directory to store the data
if not os.path.exists(f"dcopf{nbus}"):
    os.makedirs(f"dcopf{nbus}")

# %%
def get_constraints(m):
    ini_ineq_matrix = []
    ini_ineq_rhs = []
    ini_eq_matrix = []
    ini_eq_rhs = []
    a_matrix = m.getA().toarray()

    i = 0
    for constr in m.getConstrs():
        if constr.sense == '<':
            ini_ineq_matrix.append(a_matrix[i,:])
            ini_ineq_rhs.append(constr.RHS)
            i += 1
        elif constr.sense == '>':
            ini_ineq_matrix.append(-a_matrix[i,:])
            ini_ineq_rhs.append(-constr.RHS)
            i += 1
        elif constr.sense == '=':
            ini_eq_matrix.append(a_matrix[i,:])
            ini_eq_rhs.append(constr.RHS)
            i += 1

    ini_ineq_matrix = np.array(ini_ineq_matrix)
    ini_ineq_rhs = np.array(ini_ineq_rhs)
    ini_eq_matrix = np.array(ini_eq_matrix)
    ini_eq_rhs = np.array(ini_eq_rhs)

    return ini_ineq_matrix, ini_ineq_rhs, ini_eq_matrix, ini_eq_rhs

def get_objective(m):
    ini_q_matrix = m.getQ().toarray()
    ini_p_vec = []
    variables = m.getVars()
    for var in variables:
        ini_p_vec.append(var.obj)
    ini_p_vec = np.array(ini_p_vec)

    return ini_q_matrix, ini_p_vec

def get_bound(m):
    variables = m.getVars()
    ini_lb = []
    ini_ub = []
    for var in variables:
        ini_lb.append(var.lb)
        ini_ub.append(var.ub)
    ini_lb = np.array(ini_lb)
    ini_ub = np.array(ini_ub)

    return ini_lb, ini_ub

# %%
def perturb_eq_rhs(ini_eq_rhs, perturb_rate=perturb_rate):
    eq_rhs = ini_eq_rhs.copy()
    nonzero_indices = np.nonzero(ini_eq_rhs)
    nonzero_values = ini_eq_rhs[nonzero_indices]
    perturbed_values = nonzero_values * (1 + np.random.uniform(-perturb_rate, perturb_rate, size=nonzero_values.shape))
    eq_rhs[nonzero_indices] = perturbed_values
    return eq_rhs

# %%
def build_gurobi_model(Q, p, A, x, G, h, lb, ub):
    model = Model("qp")
    model.setParam('OutputFlag', 0)
    model.setParam('FeasibilityTol', 1e-4)

    n_vars = Q.shape[0]
    vars = []
    for i in range(n_vars):
        vars.append(model.addVar(lb=lb[i], ub=ub[i], vtype=GRB.CONTINUOUS, name=f"x_{i}"))

    obj = QuadExpr()
    for i in range(n_vars):
        for j in range(n_vars):
            if Q[i, j] != 0:
                obj.add(vars[i] * vars[j] * Q[i, j]* 0.5)

    for i in range(n_vars):
        if p[i] != 0:
            obj.add(vars[i] * p[i])

    model.setObjective(obj, GRB.MINIMIZE)

    for i in range(A.shape[0]):
        expr = LinExpr()
        for j in range(n_vars):
            if A[i, j] != 0:
                expr.add(vars[j] * A[i, j])
        model.addConstr(expr == x[i])
    
    for i in range(G.shape[0]):
        expr = LinExpr()
        for j in range(n_vars):
            if G[i, j] != 0:
                expr.add(vars[j] * G[i, j])
        model.addConstr(expr <= h[i])
    
    return model

# %%
def solve_osqp(Q, p, A, x, G, h, lb, ub):
    solver = osqp.OSQP()
    my_A = np.vstack([A, G, np.diag(np.ones(Q.shape[0]))])
    my_l = np.hstack([x, -np.ones(h.shape[0]) * np.inf, lb])
    my_u = np.hstack([x, h, ub])
    solver.setup(P=csc_matrix(Q), q=p, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=True, eps_prim_inf=1e-10, eps_dual_inf=1e-10, eps_abs=1e-10, eps_rel=1e-10)
    results_osqp = solver.solve()
    sol_osqp = np.array(results_osqp.x)

    return sol_osqp

# %%
G, h, A, b = get_constraints(m)
Q, p = get_objective(m)
Lb, Ub = get_bound(m)

num_var = p.shape[0]
num_eq = A.shape[0]
num_ineq = G.shape[0]

print(f"number of variables: {num_var}, number of equality constraints: {num_eq}, number of inequality constraints: {num_ineq}")

# %%
X = []
Y = []
solve_time = []
for i in range(n_samples):
    print(f"{len(X)} / {i}")
    x = perturb_eq_rhs(b)
    model = build_gurobi_model(Q, p, A, x, G, h, Lb, Ub)
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    if model.status == GRB.OPTIMAL:
        sol = [var.x for var in model.getVars()]
        Y.append(sol)
        X.append(x)
        solve_time.append(end_time - start_time)

    if len(X) % 100 == 0:
        data = {'Q':Q, 'p':p, 'A':A, 'X':X, 'G':G, 'h':h, 'Lb':Lb, 'Ub':Ub, 'Y':Y, 'solve_time':solve_time}
        with open(f"dcopf{nbus}_data", 'wb') as f:
            pickle.dump(data, f)
    if len(X) % 1000 == 0:
        data = {'Q':Q, 'p':p, 'A':A, 'X':X, 'G':G, 'h':h, 'Lb':Lb, 'Ub':Ub, 'Y':Y, 'solve_time':solve_time}
        with open(f"dcopf{nbus}/dcopf{nbus}_data_{len(X)}", 'wb') as f:
            pickle.dump(data, f)

data = {'Q':Q, 'p':p, 'A':A, 'X':X, 'G':G, 'h':h, 'Lb':Lb, 'Ub':Ub, 'Y':Y, 'solve_time':solve_time}
with open(f"dcopf{nbus}_data", 'wb') as f:
    pickle.dump(data, f)