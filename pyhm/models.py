class Model(object):
    pass

class LinearModel(Model):
    def __init__(self, a, b):
        # y = a + b x
        self.a = a
        self.b = b

    def __eval__(self, x):
        return self.a() + self.b() * x

    def loglike(self, x):
        pass

class PooledModel(Model):
    def __init__(self, var1, mu2, var2):
        # Normalize need for priors and distributions here
        self.theta = priors.NormalPrior(mu2, var2)
        self.yest  = priors.NormalPrior(self.theta, self.var1)

    def loglike(self, params):
        
