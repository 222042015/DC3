import gzip
import osqp
import pickle
import numpy as np
import os

from tqdm import tqdm
from scipy.sparse import csc_matrix
import sys
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import SimpleProblem
num_samples = 1000
# num_var = 1000
# num_ineq = 500
# num_eq = 500

num_var = 1500
num_ineq = 750
num_eq = 750

# set random seed
np.random.seed(17)

data_dir = f"/data1/jxxiong/DC3/datasets/QP_RHS_{num_var}_{num_ineq}_{num_eq}"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

refer_data = f"/data1/jxxiong/DC3/datasets/QP_RHS_{num_var}_{num_ineq}_{num_eq}_test/qp_rhs_63.gz"
with gzip.open(refer_data, 'rb') as f:
    data_dict = pickle.load(f)
Q0 = data_dict['Q']
p0 = data_dict['p'].squeeze(-1)
A0 = data_dict['A']
G0 = data_dict['G']
c0 = data_dict['c'].squeeze(-1)

# sample multiple rhs
b0 = np.random.uniform(-1, 1, size=(num_samples, num_eq, 1)).squeeze(-1)

# for i in tqdm(range(num_samples)):
#     A01 = np.concatenate((G0, A0), axis=0)
#     zl1 = np.concatenate((-np.inf*np.ones(c0.shape), b0[i, :]), axis=0)
#     zu1 = np.concatenate((c0, b0[i, :, :]), axis=0)

#     solver = osqp.OSQP()
#     solver.setup(P=csc_matrix(Q0)*2, q=p0, A=csc_matrix(A01),
#                 l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
#                 eps_dual_inf=1e-4, check_termination=1, adaptive_rho_interval=1)
#     results = solver.solve()

#     if results.info.status == 'solved':
#         data_dict = {'Q': Q0, 'p': p0, 'G': G0, 'c': c0,
#                     'A': A0, 'b': b0[i, :], 'A0': A01, 'zl': zl1, 'zu': zu1,
#                      'x': results.x, 'y': results.y}
    
#         dict_name = os.path.join(data_dir, f'qp_rhs_{i}.gz')
#         with gzip.open(dict_name, 'wb') as f:
#             pickle.dump(data_dict, f)
#     else:
#         print('Batch {} optimization failed.'.format(i))

problem = SimpleProblem(Q0, p0, A0, G0, c0, b0) # 
problem.calc_Y()

print(problem.Y.shape)
with open("/data1/jxxiong/DC3/datasets/random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, len(problem.Y)), 'wb') as f:
    pickle.dump(problem, f)