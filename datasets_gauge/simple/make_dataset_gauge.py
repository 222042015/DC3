import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from gauge_utils import SimpleProblem

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

num_var = 100
num_ineq = 50
num_eq = 50
num_examples = 10000

print(num_ineq, num_eq)
np.random.seed(17)

filepath = "/home/jxxiong/A-xjx/deeplde/datasets/simple/random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples)
with open(filepath, 'rb') as f:
    data = pickle.load(f)

Q = data.Q_np
p = data.p_np
A = data.A_np
X = data.X_np
G = data.G_np
h = data.h_np

L = np.ones((num_var))*-5
U = np.ones((num_var))*5
problem = SimpleProblem(Q, p, A, G, h, X, L, U)
problem.calc_Y()
print(len(problem.Y))
problem.remove_no_ip()
print(len(problem.Y))
with open("./random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(problem, f)


