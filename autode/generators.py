from scipy.integrate import odeint

class ODEGenerator:
    
    def __init__(self, model, y0):
        self.model = model
        self.y0 = y0
        
    def generate_orig(self, t):
        observations = odeint(self.model, self.y0, t)
        if observations[0] != self.y0:
            raise Exception("First date is not correct!")
        return observations
    
    def generate_ode(self, t):
        raise NotImplementedError("You created an ODEGenerator that has the generate method not implemented!")
        
class LinealODEGenerator(ODEGenerator):
    
    def __init__(self, a, b, y0):
        super(LinealODEGenerator, self).__init__(lambda yt, t: a * yt + b, y0) # pass arguments to constructor of super class