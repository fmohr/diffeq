import numpy as np
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint
from inspect import signature

class ODEEvaluator:
    
    # stores the data in the object
    def __init__(self, times, observations):
        if len(times) != len(observations):
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
        
        # extract parameters that must be optimized for this model
        sig = signature(model)
        params = sig.parameters
        
        # adopt tuning process
        best_model = None
        best_score = np.inf
        def residual(params):
            if len(params) > 0:
                param_vals = tuple([param.value for name, param in params.items()])
                predictions_for_func_values = odeint(model, self.observations[0], self.times, args=(param_vals))
            else:
                predictions_for_func_values = odeint(model, self.observations[0], self.times)
            return (self.observations - predictions_for_func_values).ravel()
        
        
        # try to optimize parameters in a number of iterations
        for exp in range(max_exponent):
            if best_score < self.threshold:
                break
            lower = -self.base**exp
            upper = self.base**exp
            for i in range(attempts_per_exponent):
                
                # stop condition
                if best_score < self.threshold:
                    break

                # set up randomly initialized parameters
                params_to_tune = Parameters()
                for name, param in params.items():
                    if name not in ["Y", "t"]:
                        init = np.random.random() * (upper - lower) + lower
                        params_to_tune.add(name, value = init, min=lower, max=upper)

                # fit model and find predicted values
                try:
                    result = minimize(residual, params_to_tune, method='leastsq',max_nfev = 10000)

                    # update the best fitted solution if this one is the best
                    if result.chisqr < best_score:
                        best_score = result.chisqr
                        best_model = result.params
                
                except KeyboardInterrupt: # forward interrupts
                    raise
                    
                except:
                    pass # ignore failures
                    
        self.params_for_last_model = best_model
        return best_score