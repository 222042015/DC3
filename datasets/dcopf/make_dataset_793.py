# %%
import julia 
from julia import Julia
jl = Julia(compiled_modules=False)  # Initialize Julia interpreter
Julia(runtime="/home/jxxiong/julia-1.9.2/bin/julia")
from julia import Base
# Now you can call functions from Julia's Base module
from julia import Main

# %%
import numpy as np
import gurobipy as gp

from gurobipy import Model, GRB, QuadExpr, LinExpr
from scipy.sparse import csc_matrix
import pickle
import os

# %%
case_dcit = {200: "pglib_opf_case200_activ.m",
             2000: "pglib_opf_case2000_goc.m",
             10000: "pglib_opf_case10000_goc.m",
             2312: "pglib_opf_case2312_goc.m",
             793: "pglib_opf_case793_goc.m",
             3970: "pglib_opf_case3970_goc.m"}

# %%
nbus = 3970
case_name = case_dcit[nbus]
Main.case_name = case_name
Main.nbus = nbus

# %%
Main.eval("""
using PowerModels
using JuMP
using Ipopt

file_path = "/home/jxxiong/A-xjx/DC3/datasets/dcopf/";
network_data_orig = PowerModels.parse_file(file_path*case_name);
optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0);
""")

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

# %% [markdown]
# ## Get Q, p, A, G, h

# %%
Main.eval("""
pm = instantiate_model(network_data_orig, DCPPowerModel, PowerModels.build_opf);
JuMP.write_to_file(pm.model, "dcopf$(nbus)_model.mps")
""")

# %%
m = gp.read(f'/home/jxxiong/A-xjx/DC3/datasets/dcopf/dcopf{nbus}_model.mps')
G, h, A, _ = get_constraints(m)
Q, p = get_objective(m)
Lb, Ub = get_bound(m)

# %%
print(G.shape, h.shape, A.shape, Q.shape, p.shape, Lb.shape, Ub.shape)

# %% [markdown]
# ## Generate eq_rhs (X)

# %%
num_samples = 10000
perturb = 0.3
X = []
Y = []

Main.perturb = perturb

if not os.path.exists(f"dcopf{nbus}"):
    os.makedirs(f"dcopf{nbus}")

# %%
while len(X) < num_samples:
    Main.eval("""
        network_data = deepcopy(network_data_orig)
        for (load_id, load) in network_data["load"]
            perturbation_factor = (1 - perturb) + (rand() * perturb * 2);
            load["pd"] *= perturbation_factor;
            load["qd"] *= perturbation_factor;
        end

        pm = instantiate_model(network_data, DCPPowerModel, PowerModels.build_opf);
        result = optimize_model!(pm, optimizer=optimizer);
        status = (result["termination_status"] == MOI.LOCALLY_SOLVED || result["termination_status"] == MOI.OPTIMAL);
    """)

    if Main.status:
        print(Main.status)
        Main.eval("""JuMP.write_to_file(pm.model, "dcopf$(nbus)_model.mps")""")
        m = gp.read(f'/home/jxxiong/A-xjx/DC3/datasets/dcopf/dcopf{nbus}_model.mps')
        m.optimize()
        m.setParam('OutputFlag', 0)
        if m.status == GRB.Status.OPTIMAL:
            _, _, _, x = get_constraints(m)
            X.append(x)
            Y.append([var.x for var in m.getVars()])
    
    if len(X) % 100 == 0 and len(X) != 0:
        # save the data A, X, G, h, Q, p, lb, ub
        data = {'Q':Q, 'p':p, 'A':A, 'X':X, 'G':G, 'h':h, 'Lb':Lb, 'Ub':Ub, 'Y':[]}
        with open(f"dcopf{nbus}_data", 'wb') as f:
            pickle.dump(data, f)
    if len(X) % 1000 == 0 and len(X) != 0:
        data = {'Q':Q, 'p':p, 'A':A, 'X':X, 'G':G, 'h':h, 'Lb':Lb, 'Ub':Ub, 'Y':Y}
        with open(f"dcopf{nbus}/dcopf{nbus}_data_{len(X)}", 'wb') as f:
            pickle.dump(data, f)

with open(f"dcopf{nbus}_data", 'wb') as f:
    pickle.dump(data, f)
