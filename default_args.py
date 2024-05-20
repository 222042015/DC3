def baseline_opt_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000

    if prob_type == 'simple':
        defaults['corrEps'] = 1e-4
    elif prob_type == 'nonconvex':
        defaults['corrEps'] = 1e-4
    elif 'acopf' in prob_type:
        defaults['corrEps'] = 1e-4

    return defaults

def baseline_nn_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 200
    defaults['nonconvexIneq'] = 100
    defaults['nonconvexEq'] = 100
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50

    if prob_type == 'simple':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 100
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = False
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif prob_type == 'nonconvex':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 100
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif 'acopf' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 100
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-5
        defaults['corrMomentum'] = 0.5
    else:
        raise NotImplementedError

    return defaults

def baseline_eq_nn_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50

    if prob_type == 'simple':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif prob_type == 'nonconvex':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif 'acopf' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3
        defaults['hiddenSize'] = 200
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'full'
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-5
        defaults['corrMomentum'] = 0.5
    else:
        raise NotImplementedError

    return defaults

def method_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50

    if prob_type == 'simple':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10          # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 10
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif prob_type == 'nonconvex':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10          # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 10
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif prob_type == 'acopf57':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3 # 1e-4-118
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10             # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 5
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-4           # use 1e-5 if useCompl=False, 118
        defaults['corrMomentum'] = 0.5
    elif prob_type == 'acopf118':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4 # 1e-4-118
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10             # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 5
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-5           # use 1e-5 if useCompl=False, 118
        defaults['corrMomentum'] = 0.5
    else:
        raise NotImplementedError

    return defaults


# add default args for pdl method for different problem types
def pdl_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 1

    if prob_type == "simple" or prob_type == "convex_qcqp":
        defaults['max_outer_iter'] = 10 # K
        defaults['max_inner_iter'] = 500 #L
        defaults['alpha'] = 5 #10 # alpha
        defaults['tau'] = 0.8 # tau
        defaults['rho_max'] = 5000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 500
        defaults['rho'] = 2 #0.5 # initialize
        defaults['v'] = 0 # initialize the current maximum violations
    elif prob_type == "nonconvex":
        defaults['max_outer_iter'] = 10 
        defaults['max_inner_iter'] = 500
        defaults['alpha'] = 5 #10 
        defaults['tau'] = 0.8 
        defaults['rho_max'] = 5000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 500
        defaults['rho'] = 2 #0.5 
        defaults['v'] = 0 
    elif "acopf" in prob_type:
        defaults['max_outer_iter'] = 100 
        defaults['max_inner_iter'] = 250 
        defaults['alpha'] = 2 
        defaults['tau'] = 0.8 
        defaults['rho_max'] = 10000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['rho'] = 1.0 
        defaults['v'] = 0 
    else:
        raise NotImplementedError


    return defaults

def deeplde_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50
    defaults['useCompl'] = True
    defaults['corrEps'] = 1e-4

    if prob_type == 'simple':
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3 #1e-3
        defaults['hiddenSize'] = 200
        defaults['inner_warmstart'] = 100
        defaults['inner_iter'] = 25
        defaults['outer_iter'] = 15
        defaults['beta'] = 5
        defaults['rho'] = 0.1 # 0.0001 ineq30eq70, 1e-3 ineq70eq30
        defaults['lambda'] = 0.1 # 1e-2 ineq50eq50, 0.1 ineq30eq70, 0.1 ineq70eq30
        defaults['gamma'] = 0.01 
    elif prob_type == 'nonconvex':
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3 # 1e-4-118
        defaults['hiddenSize'] = 200
        defaults['inner_warmstart'] = 100
        defaults['inner_iter'] = 25
        defaults['outer_iter'] = 15
        defaults['beta'] = 5
        defaults['rho'] = 0.0001
        defaults['lambda'] = 0.1
        defaults['gamma'] = 0.01
    elif prob_type == 'acopf57':
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3 # 1e-4-118
        defaults['hiddenSize'] = 200
        defaults['inner_warmstart'] = 100
        defaults['inner_iter'] = 25
        defaults['outer_iter'] = 15
        defaults['beta'] = 5
        defaults['rho'] = 0.1
        defaults['lambda'] = 0.1
        defaults['gamma'] = 0.01

    else:
        raise NotImplementedError

    return defaults


def gauge_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50
    defaults['useCompl'] = True
    defaults['corrEps'] = 1e-4

    if prob_type == 'simple':
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3
        defaults['hiddenSize'] = 200
        defaults['epochs'] = 3000
    elif prob_type == 'nonconvex':
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3 # 1e-4-118
        defaults['hiddenSize'] = 200
        defaults['epochs'] = 2000
    else:
        raise NotImplementedError

    return defaults