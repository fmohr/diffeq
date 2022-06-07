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
class LorenzDataGenerator(ODEGenerator):
    
    def __init__(self, a = 10, b = 8/3, r = 126.52, x1_0 = -7.69, x2_0 = -15.61, x3_0 =90.39):
        self.a = a
        self.b = b
        self.r = r
        model = lambda xs, t: [-a*xs[0] + a * xs[1], - xs[0] * xs[2] + r * xs[0] - xs[1], xs[0] * xs[1] - b * xs[2]]
        init_vals = [x1_0, x2_0, x3_0]
        super(LorenzDataGenerator, self).__init__(model, init_vals)
        
class BesselDataGenerator(ODEGenerator):
    
    def __init__(self, a = -1, b = -1, c = 1, x1_0 = -1, x2_0 = -1):
        self.a = a
        self.b = b
        self.c = c
        model = lambda xs, t: [a*xs[0]/(t+0.001) + b * xs[1] + np.exp(-t) - np.exp(-t)/(t+0.001), c * xs[0] + np.exp(-t)]
        init_vals = [x1_0, x2_0]
        super(BesselDataGenerator, self).__init__(model, init_vals)
        
class ThreeTankDataGenerator(ODEGenerator):
    
    def __init__(self, a = -3/50, b = 1/50, c = 0, d = 1/25, e = -1/20, f = 1/100, g = 0, h = 1/20, j = -1/20, x1_0 = 50, x2_0 = 50, x3_0 = 50):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h
        self.j = j
        model = lambda xs, t: [a*xs[0] + b * xs[1] + c * xs[2], d * xs[0] + e * xs[1] + f * xs[2], g * xs[0] + h * xs[1] + j * xs[2]]
        init_vals = [x1_0, x2_0, x3_0]
        super(ThreeTankDataGenerator, self).__init__(model, init_vals)
        
class PredatoryDataGenerator(ODEGenerator):
    
    def __init__(self, a = -0.16, b = 0.08, c = 4.5, d =-0.9, x1_0 = 4, x2_0 = 4):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        model = lambda xs, t: [a * xs[0] + b * xs[0] * xs[1], c*xs[1] + d * xs[0] * xs[1]]
        init_vals = [x1_0, x2_0]
        super(PredatoryDataGenerator, self).__init__(model, init_vals)
        
class NotUniqueGenerator(ODEGenerator):
    
    def __init__(self, a = 1, b = 1, c = 1, x1_0 = -1, x2_0 = 0):
        self.a = a
        self.b = b
        self.c = c
        model = lambda xs, t: [a * xs[0] + b * xs[1], c * t]
        init_vals = [x1_0, x2_0]
        super(NotUniqueDataGenerator, self).__init__(model, init_vals)
