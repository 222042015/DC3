import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse

from utils import my_hash, str_to_bool, load_data
import default_args

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--probType', type=str, default='simple',
        choices=['simple', 'nonconvex', 'acopf57'], help='problem type')
    parser.add_argument('--baseline_method', type=str, default='dc3',
        choices=['gauge', 'pdl', 'dc3'], help='baseline method')
    parser.add_argument('--simpleVar', type=int, default=1000,
        help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int, default=500,
        help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int, default=500,
        help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int, default=1000,
        help='total number of datapoints for simple problem')
    parser.add_argument('--prefix', type=str, default='/data1/jxxiong/DC3/',
        help='directory to the results')
    parser.add_argument('--testNum', type=int, default=50,
        help='number of test datapoints')
    
    args = parser.parse_args()
    args = vars(args) # change to dictionary
    baseline_method = args['baseline_method']
    test_num = args['testNum']
    model_dir = os.path.join(args['prefix'], 'baseline_result', f'SimpleProblem-{args["simpleVar"]}-{args["simpleIneq"]}-{args["simpleEq"]}-{args["simpleEx"]}', 'method_'+args['baseline_method'])
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    
    
    if baseline_method in ['dc3', 'pdl']:
    # load test dataset
        test_dir = os.path.join(args['prefix'], 'datasets', f'random_simple_dataset_var{args["simpleVar"]}_ineq{args["simpleIneq"]}_eq{args["simpleEq"]}_ex{args["testNum"]}')
        if os.path.exists(test_dir):
            with open(test_dir, 'rb') as f:
                test_data = pickle.load(f)
        else:
            test_data = load_data(os.path.join(args['prefix'], 'datasets', 'QP_RHS_{}_{}_{}'.format(args['simpleVar'], args['simpleIneq'], args['simpleEq'])), np.arange(900, 900+args['testNum']), valid_frac=0.0, test_frac=1.0)
            with open(test_dir, 'wb') as f:
                pickle.dump(test_data, f)
        for attr in dir(test_data):
            var = getattr(test_data, attr)
            if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
                try:
                    setattr(test_data, attr, var.to(DEVICE))
                except AttributeError:
                    pass
        test_data._device = DEVICE

        test_dataset = TensorDataset(test_data.testX)
        test_loader = DataLoader(test_dataset, batch_size=1) # test the instance one by one

        # load args
        with open(os.path.join(model_dir, 'args.dict'), 'rb') as f:
            args = pickle.load(f)

        # load model
        if baseline_method == 'dc3':
            from method import NNSolver, grad_steps_all
            solver = NNSolver(test_data, args)
            # load the model.dict
            solver.load_state_dict(torch.load(os.path.join(model_dir, 'solver_net.dict')))
            solver.to(DEVICE)
            solver.eval()
            # test
            c = 0
            total_time = 0
            for Xtest in test_loader:
                Xtest = Xtest[0].to(DEVICE)
                start_time = time.time()
                Ytest = solver(Xtest)
                Ycorr, steps = grad_steps_all(test_data, Xtest, Ytest, args)
                end_time = time.time()
                total_time += end_time - start_time
                c += 1
            print(f"Average time: {total_time / test_num}")
            print(c)
            Xtest = test_data.testX.to(DEVICE)
            Ytest = solver(Xtest)
            Ycorr, steps = grad_steps_all(test_data, Xtest, Ytest, args)
            print("obj_fn: ", test_data.obj_fn(Ycorr).mean().item())
            print("ineq_dist_mean: ", test_data.ineq_dist(Xtest, Ycorr).mean().item())
            print("eq_resid_mean: ", torch.mean(torch.abs(test_data.eq_resid(Xtest, Ycorr)), dim=1).mean().item())
            
        elif baseline_method == 'pdl':
            from method_pdl import Primal_NN
            solver = Primal_NN(test_data, args)
            # load the model.dict
            solver.load_state_dict(torch.load(os.path.join(model_dir, 'primal_net.dict')))
            solver.to(DEVICE)
            solver.eval()
            # test
            total_time = 0
            for Xtest in test_loader:
                Xtest = Xtest[0].to(DEVICE)
                start_time = time.time()
                Ytest = solver(Xtest)
                end_time = time.time()
                total_time += end_time - start_time
            print(f"Average time: {total_time / test_num}")

            Xtest = test_data.testX.to(DEVICE)
            Ytest = solver(Xtest)
            print("obj_fn: ", test_data.obj_fn(Ytest).mean().item())
            print("ineq_dist_mean: ", test_data.ineq_dist(Xtest, Ytest).mean().item())
            print("eq_resid_mean: ", torch.mean(torch.abs(test_data.eq_resid(Xtest, Ytest)), dim=1).mean().item())
        
        
    elif baseline_method == 'gauge':
        
        test_dir = os.path.join(args['prefix'], 'datasets', f'random_simple_dataset_var{args["simpleVar"]}_ineq{args["simpleIneq"]}_eq{args["simpleEq"]}_ex{args["testNum"]}_bounded')
        if os.path.exists(test_dir):
            with open(test_dir, 'rb') as f:
                test_data = pickle.load(f)
        else:
            raise FileNotFoundError(f"Test dataset {test_dir} does not exist.")

        for attr in dir(test_data):
            var = getattr(test_data, attr)
            if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
                try:
                    setattr(test_data, attr, var.to(DEVICE))
                except AttributeError:
                    pass
        test_data._device = DEVICE

        test_dataset = TensorDataset(test_data.testX, test_data.testIP)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # test the instance one by one

        # load args
        with open(os.path.join(model_dir, 'args.dict'), 'rb') as f:
            args = pickle.load(f)
        from method_gauge import NNSolver
        solver = NNSolver(test_data, args)
        # load the model.dict
        solver.load_state_dict(torch.load(os.path.join(model_dir, 'solver_net.dict')))
        solver.to(DEVICE)
        solver.eval()
        
        
        total_time = 0
        c = 0
        for Xtest, IPtest in test_loader:
            c += 1
            Xtest = Xtest.to(DEVICE)
            IPtest = IPtest.to(DEVICE)
            start_time = time.time()
            v_test = solver(Xtest, IPtest)
            Ypartial_test = test_data.gauge_map(v_test, IPtest, Xtest)
            Yhat_test = test_data.complete_partial(Xtest, Ypartial_test)
            end_time = time.time()
            total_time += end_time - start_time
        print(f"Average time: {total_time / test_num}")
        print(c)


        # run in batch to evaluate the accuracy 
        Xtest = test_data.testX.to(DEVICE)
        IPtest = test_data.testIP.to(DEVICE)
        v_test = solver(Xtest, IPtest)
        Ypartial_test = test_data.gauge_map(v_test, IPtest, Xtest)
        Yhat_test = test_data.complete_partial(Xtest, Ypartial_test)
        print("obj_fn: ", test_data.obj_fn(Yhat_test).mean().item())
        print("ineq_dist_mean: ", test_data.ineq_dist(Xtest, Yhat_test).mean().item())
        print("eq_resid_mean: ", torch.mean(torch.abs(test_data.eq_resid(Xtest, Yhat_test)), dim=1).mean().item())

if __name__ == '__main__':
    main()