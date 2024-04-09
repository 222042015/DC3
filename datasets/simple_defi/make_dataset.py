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

num_var = 100
num_ineq = 50
num_eq_list = [50]
num_examples = 9000


# num_ineq = 50
# for num_eq in [10, 30, 50, 70, 90]:
for num_eq in num_eq_list:
    print(num_ineq, num_eq)
    np.random.seed(17)
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
    G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)

    A = np.zeros((num_eq, num_var))

    # Fill the first (num_eq - 1) columns of A with independent vectors
    # We ensure that these columns are linearly independent
    for i in range(num_eq + 5):
        A[:, i] = np.random.normal(loc=0, scale=1., size=num_eq)

    # Fill the remaining columns of A with linear combinations of the first (num_eq - 1) columns
    for i in range(num_eq + 5, num_var):
        coefficients = np.random.random(num_eq - 1)
        A[:, i] = np.dot(A[:, :num_eq - 1], coefficients)

    problem = SimpleProblem(Q, p, A, G, h, X, test_frac=0.001)
    problem.calc_Y()
    print(len(problem.Y))

    with open("./random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
        pickle.dump(problem, f)

# num_eq = 50
# for num_ineq in [10, 30, 70, 90]:
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
