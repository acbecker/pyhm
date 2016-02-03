import numpy as np
import scipy.stats as stats

class Prior(object):
    def __init___(self):
        self.distribution = stats.norm
    def loc(self):
        return 0.0
    def scale(self):
        return 1.0

    def rvs(self, size):
        return self.distribution(loc=self.loc(), scale=self.scale()).rvs(size=size)
    def lnlike(self, x):
        return self.distribution.logpdf(x, loc=self.loc(), scale=self.scale())
        
       
class UniformPrior(Prior):
    def __init__(self, min, max):
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform
        self.min = min # Parameter
        self.max = max # Parameter
        self.distribution = stats.uniform

    def loc(self):
        return self.min
    def scale(self):
        return self.max-self.min

class NormalPrior(Prior):
    def __init__(self, mu, var):
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm
        self.mu  = mu  # Parameter
        self.var = var # Parameter
        self.distribution = stats.norm

    def loc(self):
        return self.mu()
    def scale(self):
        return np.sqrt(self.var())

class InverseGammaPrior(Prior):
    def __init__(self, alpha, beta):
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html#scipy.stats.invgamma
        self.alpha = alpha
        self.beta = beta
        self.distributon = stats.invgamma

    def loc(self):
        return self.alpha()
    def scale(self):
        return self.beta()

    # Relate to inverse chi2 distribution parameters
    def invChi2Scale(self):
        return np.sqrt(self.beta() / self.alpha())
    def invChi2Dof(self):
        return 2.0 * self.alpha()

class HalfCauchyPrior(Prior):
    def __init__(self, loc, scale):
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfcauchy.html#scipy.stats.halfcauchy
        self.l = loc
        self.s = scale
        self.distribution = stats.halfcauchy

    def loc(self):
        return self.l
    def scale(self):
        return self.s

class ExponentialPrior(Prior):
    def __init__(self, scale):
        self.l = 0
        self.s = scale
        self.distribution = stats.expon

    def loc(self):
        return self.l
    def scale(self):
        return self.s

class BetaPrior(Prior):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.distribution = stats.beta

    def loc(self):
        return self.l
    def scale(self):
        return self.s
