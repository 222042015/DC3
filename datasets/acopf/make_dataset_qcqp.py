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
import scipy

# %%
case_dcit = {200: "pglib_opf_case200_activ.m",
             2000: "pglib_opf_case2000_goc.m",
             10000: "pglib_opf_case10000_goc.m",
             2312: "pglib_opf_case2312_goc.m",
             793: "pglib_opf_case793_goc.m",
             3970: "pglib_opf_case3970_goc.m",
             57: "pglib_opf_case57_ieee.m",
             3: "pglib_opf_case3_lmbd.m"}

# %%
nbus = 57
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
def get_lin_eq_constr(m):
    eq_lin_constrs = [c for c in m.getConstrs() if c.sense == '='] 
    eq_lin_matrix = []
    eq_lin_rhs = []
    a_matrix = m.getA().toarray()
    for i, constr in enumerate(eq_lin_constrs):
        if constr.sense == '=':
            eq_lin_matrix.append(a_matrix[i,:])
            eq_lin_rhs.append(constr.RHS)
    eq_lin_matrix = np.array(eq_lin_matrix)
    eq_lin_rhs = np.array(eq_lin_rhs)
    
    return eq_lin_matrix, eq_lin_rhs

def get_quad_eq_constr(m):
    eq_quad_constrs = [qc for qc in m.getQConstrs() if qc.qcsense == '=']
    num_var = len(m.getVars())
    var_index = {v.VarName: i for i, v in enumerate(m.getVars())}
    eq_quad_H = []
    eq_quad_g = []
    eq_quad_h = []

    for qc in eq_quad_constrs:
        print(qc)
        H_row = []
        H_col = []
        H_data = []

        g_tmp = np.zeros(num_var)
        if qc.qcsense == '=':
            expr = m.getQCRow(qc)
            print(expr)
            # Quadratic terms
            for i in range(expr.size()):
                var1 = expr.getVar1(i)
                var2 = expr.getVar2(i)
                coeff = expr.getCoeff(i)
                idx1 = var_index[var1.VarName]
                idx2 = var_index[var2.VarName]
                if idx1 == idx2:
                    H_row.append(idx1)
                    H_col.append(idx2)
                    H_data.append(coeff/2)
                else:
                    H_row.append(idx1)
                    H_col.append(idx2)
                    H_data.append(coeff/4)
                    H_row.append(idx2)
                    H_col.append(idx1)
                    H_data.append(coeff/4)

            print(H_row)
            print(H_col)
            print(H_data)
            # Linear part of the quadratic constraint
            lin_expr = expr.getLinExpr()
            eq_quad_h.append(qc.getAttr('qcrhs'))
            for j in range(lin_expr.size()):
                var = lin_expr.getVar(j)
                coeff = lin_expr.getCoeff(j)
                g_tmp[var_index[var.VarName]] = coeff
            print(qc.getAttr('qcrhs'))
            print(g_tmp)

            H_tmp = csc_matrix((np.array(H_data), (np.array(H_row), np.array(H_col))), shape=(num_var, num_var))
            eq_quad_H.append(H_tmp.toarray())
            eq_quad_g.append(g_tmp)

    eq_quad_H = np.array(eq_quad_H)
    eq_quad_g = np.array(eq_quad_g)
    eq_quad_h = np.array(eq_quad_h)
    
    return eq_quad_H, eq_quad_g, eq_quad_h

def get_quad_ineq_constr(m):
    ineq_quad_constrs = [qc for qc in m.getQConstrs() if qc.qcsense != '=']
    ineq_quad_H = []
    ineq_quad_g = []
    ineq_quad_h = []
    num_var = len(m.getVars())
    var_index = {v.VarName: i for i, v in enumerate(m.getVars())}

    for qc in ineq_quad_constrs:
        H_row = []
        H_col = []
        H_data = []

        g_tmp = np.zeros(num_var)
        if qc.qcsense == '<':
            expr = m.getQCRow(qc)
            # print(expr)
            # Quadratic terms
            for i in range(expr.size()):
                var1 = expr.getVar1(i)
                var2 = expr.getVar2(i)
                coeff = expr.getCoeff(i)
                idx1 = var_index[var1.VarName]
                idx2 = var_index[var2.VarName]
                if idx1 == idx2:
                    H_row.append(idx1)
                    H_col.append(idx2)
                    H_data.append(coeff / 2)
                else:
                    H_row.append(idx1)
                    H_col.append(idx2)
                    H_data.append(coeff/4)
                    H_row.append(idx2)
                    H_col.append(idx1)
                    H_data.append(coeff/4)

            # Linear part of the quadratic constraint
            lin_expr = expr.getLinExpr()
            ineq_quad_h.append(qc.getAttr('qcrhs'))
            # print(qc.getAttr('qcrhs'))
            for j in range(lin_expr.size()):
                var = lin_expr.getVar(j)
                coeff = lin_expr.getCoeff(j)
                # G[var_index[var.VarName]] += coeff
                g_tmp[var_index[var.VarName]] = coeff

            H_tmp = csc_matrix((np.array(H_data), (np.array(H_row), np.array(H_col))), shape=(num_var, num_var))
            ineq_quad_H.append(H_tmp.toarray())
            ineq_quad_g.append(g_tmp)
        elif qc.qcsense == '>':
            expr = m.getQCRow(qc)
            # print(expr)
            # Quadratic terms
            for i in range(expr.size()):
                var1 = expr.getVar1(i)
                var2 = expr.getVar2(i)
                coeff = expr.getCoeff(i)
                idx1 = var_index[var1.VarName]
                idx2 = var_index[var2.VarName]
                if idx1 == idx2:
                    H_row.append(idx1)
                    H_col.append(idx2)
                    H_data.append(-coeff / 2)
                else:
                    H_row.append(idx1)
                    H_col.append(idx2)
                    H_data.append(-coeff/4)
                    H_row.append(idx2)
                    H_col.append(idx1)
                    H_data.append(-coeff/4)
            # Linear part of the quadratic constraint
            lin_expr = expr.getLinExpr()
            ineq_quad_h.append(-qc.getAttr('qcrhs'))
            # print(-qc.getAttr('qcrhs'))
            for j in range(lin_expr.size()):
                var = lin_expr.getVar(j)
                coeff = lin_expr.getCoeff(j)
                g_tmp[var_index[var.VarName]] = -coeff

            H_tmp = csc_matrix((np.array(H_data), (np.array(H_row), np.array(H_col))), shape=(num_var, num_var))
            ineq_quad_H.append(H_tmp.toarray())
            ineq_quad_g.append(g_tmp)

    ineq_quad_H = np.array(ineq_quad_H)
    ineq_quad_g = np.array(ineq_quad_g)
    ineq_quad_h = np.array(ineq_quad_h)
    
    return ineq_quad_H, ineq_quad_g, ineq_quad_h

def get_objective(m):
    Q = m.getQ().toarray()
    p = []
    variables = m.getVars()
    for var in variables:
        p.append(var.obj)
    p = np.array(p)
    
    return Q, p

def get_bound(m):
    Lb = np.array([v.lb for v in m.getVars()])
    Ub = np.array([v.ub for v in m.getVars()])

    return Lb, Ub

# %% [markdown]
# ## Get Q, p, A, G, h

# %%
Main.eval("""
pm = instantiate_model(network_data_orig, ACRPowerModel, PowerModels.build_opf);
JuMP.write_to_file(pm.model, "acopf$(nbus)_model.mps")
""")

# %%
m = gp.read(f'/home/jxxiong/A-xjx/DC3/datasets/acopf/acopf{nbus}_model.mps')
# G, h, A, _ = get_constraints(m)
Q, p = get_objective(m)
Lb, Ub = get_bound(m)
num_var = len(m.getVars())

ineq_quad_H, ineq_quad_g, ineq_quad_h = get_quad_ineq_constr(m)
eq_quad_H, eq_quad_g, eq_quad_h = get_quad_eq_constr(m)
eq_lin_matrix, eq_lin_rhs = get_lin_eq_constr(m)

eq_H = np.concatenate([np.zeros((len(eq_lin_matrix), num_var, num_var)), eq_quad_H], axis=0)
eq_g = np.concatenate([eq_lin_matrix, eq_quad_g], axis=0)
# eq_h = np.concatenate([eq_lin_rhs, eq_quad_h], axis=0)

# %%
print(eq_quad_H.shape, eq_quad_g.shape, eq_quad_h.shape)
print(ineq_quad_H.shape, ineq_quad_g.shape, ineq_quad_h.shape)
print(eq_lin_matrix.shape, Q.shape, p.shape, Lb.shape, Ub.shape)

# %% [markdown]
# ## Generate eq_rhs (X)

# %%
num_samples = 10000
perturb = 0.2
eq_h = []

Main.perturb = perturb

if not os.path.exists(f"acopf{nbus}"):
    os.makedirs(f"acopf{nbus}")

# %%
while len(eq_h) < num_samples:
    Main.eval("""
        network_data = deepcopy(network_data_orig)
        for (load_id, load) in network_data["load"]
            perturbation_factor = (1 - perturb) + (rand() * perturb * 2);
            load["pd"] *= perturbation_factor;
            load["qd"] *= perturbation_factor;
        end

        pm = instantiate_model(network_data, ACRPowerModel, PowerModels.build_opf);
        result = optimize_model!(pm, optimizer=optimizer);
        status = (result["termination_status"] == MOI.LOCALLY_SOLVED || result["termination_status"] == MOI.OPTIMAL);
    """)

    if Main.status:
        Main.eval("""JuMP.write_to_file(pm.model, "acopf$(nbus)_model.mps")""")
        m = gp.read(f'/home/jxxiong/A-xjx/DC3/datasets/acopf/acopf{nbus}_model.mps')
        _, eq_lin_rhs = get_lin_eq_constr(m)
        eq_h.append(np.concatenate([eq_lin_rhs, eq_quad_h], axis=0))
    
    if len(eq_h) % 1000 == 0 and len(eq_h) != 0:
        # save the data A, X, G, h, Q, p, lb, ub
        data = {'Q':Q, 'p':p, 'eq_H':eq_H, 'eq_g':eq_g, 'eq_h':eq_h, 'ineq_H':ineq_quad_H, 'ineq_g': ineq_quad_g, 'ineq_h': ineq_quad_h, 'Lb':Lb, 'Ub':Ub, 'Y':[]}
        with open(f"acopf{nbus}_data", 'wb') as f:
            pickle.dump(data, f)

with open(f"acopf{nbus}_data", 'wb') as f:
    pickle.dump(data, f)

# %%

scipy.io.savemat('acopf{}_data.mat'.format(nbus), data)