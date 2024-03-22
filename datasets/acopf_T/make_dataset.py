import os
import pickle
import numpy as np
import sys
import scipy.io
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
# from utils import ACOPFProblem
# [3,5,24,30,73, 118, 200, 300, 500, 1354, 2000
nbus = 39
num = 1200

data = scipy.io.loadmat('./matlab_datasets/data/ACOPF_01_variation/FeasiblePairs_case{}.mat'.format(nbus))
ppc_mat = scipy.io.loadmat('./matlab_datasets/data/ACOPF_01_variation/case{}.mat'.format(nbus))
ppc =   {'version': int(ppc_mat['my_model']['version'][0,0]), \
        'baseMVA': float(ppc_mat['my_model']['baseMVA'][0,0]), \
        'bus':ppc_mat['my_model']['bus'][0,0], \
        'gen':ppc_mat['my_model']['gen'][0,0], \
        'branch':ppc_mat['my_model']['branch'][0,0], \
        'gencost':ppc_mat['my_model']['gencost'][0,0]}
data['ppc'] = ppc

np.random.seed(1111)
sample_index = np.random.choice([i for i in range(data/['Dem'].T.shape[0])], num, replace=False)
# sample_index = np.arange(num)
data['Dem'] = data['Dem'].T[sample_index, :]
data['Gen'] = data['Gen'].T[sample_index, :]
data['Vol'] = data['Vol'].T[sample_index, :]
data['Ybus'] = data['Ybus']
data['sample'] = 'truncated_normal'

# with open("./datasets/acopf/acopf_2023_{}_{}_dataset".format(nbus, num), 'wb') as f:
#     pickle.dump(data, f)

with open("./acopf{}_dataset".format(nbus), 'wb') as f:
    pickle.dump(data, f)
