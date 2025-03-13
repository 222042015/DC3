import numpy as np
import pickle
import torch
import argparse
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from gauge_utils import SimpleProblem

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

# num_var = 1500
# num_ineq = 750
# num_eq = 750
# num_examples = 50
parser = argparse.ArgumentParser(description='DC3')
parser.add_argument('--numVar', type=int, default=1000)
parser.add_argument('--numIneq', type=int, default=500)
parser.add_argument('--numEq', type=int, default=500)
parser.add_argument('--numExamples', type=int, default=1000)
args = parser.parse_args()

num_var = args.numVar
num_ineq = args.numIneq
num_eq = args.numEq
num_examples = args.numExamples
if num_examples == 50:
    valid_frac = 0.0
    test_frac = 1.0
else:
    valid_frac = 0.03
    test_frac = 0.03
np.random.seed(17)

filepath = "/data1/jxxiong/DC3/datasets/random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples)
with open(filepath, 'rb') as f:
    data = pickle.load(f)

Q = data.Q_np
p = data.p_np
A = data.A_np
X = data.X_np
G = data.G_np
h = data.h_np
Y = data.Y_np

print(num_var, num_ineq, num_eq, num_examples)
L = np.ones((num_var))*-5
U = np.ones((num_var))*5
problem = SimpleProblem(Q, p, A, G, h, X, L, U, valid_frac=valid_frac, test_frac=test_frac)
problem.set_Y(Y)
# problem.calc_Y()
# print(len(problem.Y))
problem.remove_no_ip()
print(len(problem.Y))
with open("/data1/jxxiong/DC3/datasets/random_simple_dataset_var{}_ineq{}_eq{}_ex{}_bounded".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(problem, f)
print(len(problem.testY))

