import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import NonconvexProblem
import scipy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

num_var = 200
num_ineq = 100
num_eq = 100
num_examples = 10000
print(num_ineq, num_eq)

filepath = "/home/jxxiong/A-xjx/deeplde/datasets/nonconvex/random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}.mat".format(num_var, num_ineq, num_eq, num_examples)
data = scipy.io.loadmat(filepath)
np.random.seed(17)
Q = data["Q"]
p = data["p"].squeeze()
A = data["A"]
X = data["X"]
G = data["G"]
h = data["h"].squeeze()

print(Q.shape)
print(p.shape)
print(A.shape)
print(X.shape)
print(G.shape)
print(h.shape)

problem = NonconvexProblem(Q, p, A, G, h, X)
problem.calc_Y()
print(len(problem.Y))

with open("./random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(problem, f)