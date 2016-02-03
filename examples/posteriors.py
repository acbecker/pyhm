import numpy as np

class Summary(object):
    def __init__(self, chain):
        self.nwalkers, self.nstep, self.ndim = chain.shape
        self.chain = chain

    def summarize(self):
        for i in range(self.ndim):
            d = self.chain[:,:,i]
            print "%d : %.4f %.4f %.4f" % (i, np.percentile(d, 25), np.percentile(d, 50), np.percentile(d, 75))

class Convergence(object):
    def __init__(self, chain):
        pass
    

    
