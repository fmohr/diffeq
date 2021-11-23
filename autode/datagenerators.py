import numpy as np
from scipy.integrate import odeint

class ODEGenerator:
    
    def __init__(self, model, x0):
        self.model = model
        self.x0 = x0
        
    def generate_orig(self, t):
        observations = odeint(self.model, self.x0, t)
        if np.any(observations[0] != self.x0):
            raise Exception(f"First date is not correct! Expected {self.x0} but saw {observations[0]}")
        return observations
    
    def generate_ode(self, t):
        raise NotImplementedError("You created an ODEGenerator that has the generate method not implemented!")

class TankDataGenerator(ODEGenerator):
    
    def __init__(self, a = -2/25, b = 1/50, c = 2/25, d = -2/25, x1_0 = 25, x2_0 = 0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        model = lambda xs, t: [a*xs[0] + b * xs[1], c * xs[0] + d * xs[1]]
        init_vals = [x1_0, x2_0]
        super(TankDataGenerator, self).__init__(model, init_vals)