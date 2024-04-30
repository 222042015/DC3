import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import SimpleProblem

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

num_var = 200
num_ineq = 100
num_eq = 100
num_examples = 10000


num_ineq = 100
for num_eq in [20, 60, 100, 140, 180]:
    print(num_ineq, num_eq)
    np.random.seed(17)
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
    G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)

    problem = SimpleProblem(Q, p, A, G, h, X)
    problem.calc_Y()
    print(len(problem.Y))

    with open("./random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
        pickle.dump(problem, f)

# num_eq = 30
# # for num_ineq in [10, 30, 70, 90]:
# for num_ineq in [70]:
#     print(num_ineq, num_eq)
#     np.random.seed(17)
#     Q = np.diag(np.random.random(num_var))
#     p = np.random.random(num_var)
#     A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
#     X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
#     G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
#     h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)

#     problem = SimpleProblem(Q, p, A, G, h, X)
#     problem.calc_Y()
#     print(len(problem.Y))

#     with open("./random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
#         pickle.dump(problem, f)
