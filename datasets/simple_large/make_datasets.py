import gzip
import osqp
import pickle
import numpy as np

from tqdm import tqdm
from scipy.sparse import csc_matrix

num_samples = 1000
# num_var = 1000
# num_ineq = 500
# num_eq = 500

num_var = 1500
num_ineq = 750
num_eq = 750


Q0 = 0.5*np.diag(np.random.random(num_var))
p0 = np.random.random(size=(num_var, 1))
A0 = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
b0 = np.random.uniform(-1, 1, size=(num_samples, num_eq, 1))
G0 = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
c0 = np.sum(np.abs(G0@np.linalg.pinv(A0)), axis=1).reshape((num_ineq, 1))

for i in tqdm(range(num_samples)):
    A01 = np.concatenate((G0, A0), axis=0)
    zl1 = np.concatenate((-np.inf*np.ones(c0.shape), b0[i, :]), axis=0)
    zu1 = np.concatenate((c0, b0[i, :, :]), axis=0)

    solver = osqp.OSQP()
    solver.setup(P=csc_matrix(Q0)*2, q=p0, A=csc_matrix(A01),
                l=zl1, u=zu1, verbose=False, eps_prim_inf=1e-4,
                eps_dual_inf=1e-4, check_termination=1, adaptive_rho_interval=1)
    results = solver.solve()

    if results.info.status == 'solved':
        data_dict = {'Q': Q0, 'p': p0, 'G': G0, 'c': c0,
                    'A': A0, 'b': b0[i, :], 'A0': A01, 'zl': zl1, 'zu': zu1,
                     'x': results.x, 'y': results.y}
    
        dict_name = '/data1/jxxiong/DC3/datasets/QP_RHS_{}_{}_{}/qp_rhs_{}.gz'.format(num_var, num_ineq, num_eq, i)
        with gzip.open(dict_name, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        print('Batch {} optimization failed.'.format(i))