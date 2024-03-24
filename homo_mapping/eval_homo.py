import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import time
import os
from training_all import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
defaults = config()


def main():
    args = config()
    # for prob in ['qp']:
    #     if prob == 'acopf':
    #         for size in [[30, 10000], 
    #                      [118, 20000]]:
    #             for proj in ['H_Bis']:
    #                 args['opfSize'] = size
    #                 args['projType'] = proj
    #                 args['probType'] = prob
    #                 test_single(args)
    #     else:
    #         for size in [[200, 100, 100, 20000]]:
    #             for proj in ['H_Bis']:
    #                 args['probSize'] = size
    #                 args['projType'] = proj
    #                 args['probType'] = prob
    #                 test_single(args)
    # args['opfSize'] = [118, 20000]
    # args['projType'] = 'D_Proj'
    # args['probType'] = 'acopf'

    args['probType'] = 'nonconvex'
    args['projType'] = 'H_Bis'
    args['probSize'] = [100, 50, 50, 10000]
    args['testSize'] = 833
    test_single(args)

def test_single(args):
    data, result_save_dir, model_save_dir = load_instance(args)
    test_nn_solver(data, args, model_save_dir, result_save_dir)

def test_nn_solver(data, args, model_save_dir, result_save_dir):
    print(args['probType'], args['projType'])
    args['proj_para']['useTestCorr'] = True
    DEVICE = torch.device("cuda")
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE
    Xtest = data.testX.to(DEVICE)
    Ytest = data.testY.squeeze().to(DEVICE)

    homeo_mapping = torch.load(os.path.join(model_save_dir, 'mapping.pth'), map_location=DEVICE)
    solver_net = torch.load(os.path.join(model_save_dir, 'solver_net.pth'), map_location=DEVICE)
    epoch_stats = {}
    solver_net.eval()
    eval_solution(data, Xtest, Ytest, solver_net, homeo_mapping, args, 'test', epoch_stats)
    with open(os.path.join(result_save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(epoch_stats, f)


# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value

# Modifies stats in place

def eval_solution(data, X, Ytarget, solver_net, homeo_mapping, args, prefix, stats):
    solver_net.eval()
    homeo_mapping.eval()
    ### NN solution prediction
    raw_start_time = time.time()
    with torch.no_grad():
        Y_pred = solver_net(X)
        Y_pred_scale = data.scale(X, Y_pred)
        if 'Eq' in args['algoType']:
            Y = data.complete_partial(X, Y_pred_scale, backward=False)
        else:
            Y = Y_pred_scale
    raw_end_time = time.time()
    NN_pred_time = raw_end_time - raw_start_time

    print("nn time", NN_pred_time)

    ### Post-processing for infeasible only
    steps = args['proj_para']['corrTestMaxSteps']
    eps_converge = args['proj_para']['corrEps']
    violation = data.check_feasibility(X, Y)
    penalty = torch.max(torch.abs(violation), dim=1)[0]
    infeasible_index = (penalty > eps_converge).view(-1)
    Y_pred_infeasible = Y[infeasible_index]
    num_infeasible_prediction = Y_pred_infeasible.shape[0]
    Ycorr = Y.detach().clone()
    print(f'num of infeasible instance {Y_pred_infeasible.shape[0]}')
    if num_infeasible_prediction > 0:
        cor_start_time = time.time()
        if args['proj_para']['useTestCorr']:
            if 'H_Bis' in args['algoType']:
                Yproj, steps = homeo_bisection(homeo_mapping, data, args, Y_pred[infeasible_index], X[infeasible_index])
            elif 'G_Bis' in args['algoType']:
                Yproj, steps = gauge_bisection(homeo_mapping, data, args, Y_pred[infeasible_index], X[infeasible_index])
            elif 'D_Proj' in args['algoType']:
                Yproj, steps = diff_projection(data, X[infeasible_index], Y[infeasible_index], args)
            elif 'Proj' in args['algoType']:
                Yproj = data.opt_proj(X[infeasible_index], Y[infeasible_index]).to(Y.device).view(
                    Y_pred_infeasible.shape)
            elif 'WS' in args['algoType']:
                Yproj = data.opt_warmstart(X[infeasible_index], Y[infeasible_index]).to(Y.device).view(
                    Y_pred_infeasible.shape)
            else:
                Yproj = Y_pred_infeasible
            Ycorr[infeasible_index] = Yproj
        cor_end_time = time.time()
        Proj_time = cor_end_time - cor_start_time
    else:
        Proj_time = 0.0

    make_prefix = lambda x: "{}_{}".format(prefix, x)
    dict_agg(stats, make_prefix('time'), Proj_time + NN_pred_time, op='sum')
    # dict_agg(stats, make_prefix('proj_time'), Proj_time, op='sum')
    # dict_agg(stats, make_prefix('raw_time'), NN_pred_time, op='sum')
    dict_agg(stats, make_prefix('steps'), np.array([steps]))
    # dict_agg(stats, make_prefix('loss'), total_loss(data, X, Ynew, args).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_resid(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_resid(X, Ycorr), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_0'),
             torch.sum(data.ineq_resid(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_1'),
             torch.sum(data.ineq_resid(X, Ycorr) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_2'),
             torch.sum(data.ineq_resid(X, Ycorr) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_0'),
             torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_1'),
             torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_2'),
             torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_time'), (raw_end_time-end_time) + (base_end_time-start_time), op='sum')
    # dict_agg(stats, make_prefix('raw_eval'), data.obj_fn(Ynew).detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(data.ineq_dist(X, Ynew), dim=1)[0].detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(data.ineq_dist(X, Ynew), dim=1).detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_ineq_num_viol_0'),
    #          torch.sum(data.ineq_dist(X, Ynew) > eps_converge, dim=1).detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_ineq_num_viol_1'),
    #          torch.sum(data.ineq_dist(X, Ynew) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_ineq_num_viol_2'),
    #          torch.sum(data.ineq_dist(X, Ynew) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_eq_max'),
    #          torch.max(torch.abs(data.eq_resid(X, Ynew)), dim=1)[0].detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_eq_mean'),
    #          torch.mean(torch.abs(data.eq_resid(X, Ynew)), dim=1).detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
    #          torch.sum(torch.abs(data.eq_resid(X, Ynew)) > eps_converge, dim=1).detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
    #          torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    # dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
    #          torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 100 * eps_converge, dim=1).detach().cpu().numpy())


    for key in stats.keys():
        stats[key] = np.expand_dims(np.array(stats[key]), axis=0)
    
    stats['valid_time'] = np.ones(1000)
    return stats


if __name__ == '__main__':
    main()