import numpy as np
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint
from inspect import signature
import matplotlib.pyplot as plt

class ODEEvaluator:
    
    # stores the data in the object
    def __init__(self, times, observations):
        if type(observations) != np.ndarray:
            raise ValueError(f"Observations must be a numpy array but are {type(observations)}")
        if len(observations.shape) != 2:
            raise ValueError(f"Observations must be a 2D numpy array. Current shape is however {observations.shape}")
        if len(times) != observations.shape[0]:
            raise ValueError("List of timestamps and list of values should have same length!")
        self.times = times
        self.observations = observations
        
    # evaluates a concrete model
    def evaluate(self, model):
        raise NotImplementedError("You created an evaluator that has the evalute method not implemented!")
        



class ODEINTEvaluator(ODEEvaluator):
    
    def __init__(self, times, observations, threshold = 10**-5, base = 10):
        super(ODEINTEvaluator, self).__init__(times, observations) # pass times and observations to super class
        self.threshold = threshold
        self.base = base
    
    def evaluate(self, model, attempts_per_exponent=100, max_exponent=10):
        
        # if we get just one model, implicitly create a list of models (containing only one)
        if type(model) != list:
            model = [model]
        
        # extract parameters that must be optimized for this model
        params_all = {}
        params_per_model = []
        for f in model:
            params_for_this_model = signature(f).parameters
            params_per_model.append(params_for_this_model)
            for k, v in params_for_this_model.items():
                params_all[k] = v
                
        # get names of parameters that are subject to optimization
        names_of_params_to_be_optimized = [p for p in params_all if not p in ["t"] + ["x" + str(i) for i in range(0, 100)]]
        
        # adopt tuning process
        best_model = None
        best_score = np.inf
        def residual(params):
                        
            def internal_model(*args):
                
                # by definition of odeint, the first two arguments are y and t
                func_vals = args[0]
                t = args[1]
                param_vals = args[2:]
                
                # now compute the output of each equation of the model given the parameters
                out = []
                for i, f in enumerate(model):
                    params_for_this_model = {}
                    for param_name in params_per_model[i]:
                        if param_name in names_of_params_to_be_optimized:
                            params_for_this_model[param_name] = param_vals[names_of_params_to_be_optimized.index(param_name)]
                        elif param_name == "t":
                            params_for_this_model["t"] = t
                        else:
                            param_set = False
                            for j in range(1, len(func_vals) + 1):
                                if param_name == "x" + str(j):
                                    params_for_this_model[param_name] = func_vals[j - 1]
                                    param_set = True
                                    break
                            if not param_set:
                                raise Exception(f"Could not set param {param_name}")
                    
                    # evaluate equation and append the result to the output list
                    out.append(f(**params_for_this_model))
                return out
            
            if len(params) > 0:
                param_vals = tuple([param.value for name, param in params.items()])
                predictions_for_func_values = odeint(internal_model, self.observations[0], self.times, args=(param_vals))
            else:
                predictions_for_func_values = odeint(internal_model, self.observations[0], self.times)
            return (self.observations - predictions_for_func_values).ravel()
        
        
        # try to optimize parameters in a number of iterations
        for exp in range(max_exponent):
            if best_score < self.threshold:
                break
            lower = -self.base**exp
            upper = self.base**exp
            #print(f"Setting exponent to {exp}")
            for i in range(attempts_per_exponent):
                
                #print(f"Trying {i}-th optimization for exp {exp}")
                
                # stop condition
                if best_score < self.threshold:
                    break

                # set up randomly initialized parameters
                params_to_tune = Parameters()
                for name, param in params_all.items():
                    if name in names_of_params_to_be_optimized:
                        init = np.random.random() * (upper - lower) + lower
                        params_to_tune.add(name, value = init, min=lower, max=upper)

                # fit model and find predicted values
                try:
                    result = minimize(residual, params_to_tune, method='leastsq',max_nfev = 10**2)

                    # update the best fitted solution if this one is the best
                    if result.chisqr < best_score:
                        best_score = result.chisqr
                        best_model = result.params
                        print(f"Found new best model with score {best_score}")
                
                except KeyboardInterrupt: # forward interrupts
                    raise
                    
                except:
                    pass # ignore failures
                    #raise
                    
        self.params_for_last_model = best_model
        return best_score