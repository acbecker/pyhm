import numpy as np

class Summary(object):
    def summarize(self, data):
        nwalkers, nstep, ndim = data.shape
        for i in range(ndim):
            p25 = np.percentile(data[:,:,i], 25)
            p50 = np.percentile(data[:,:,i], 50)
            p75 = np.percentile(data[:,:,i], 75)

            print "%s  : %.4f %.4f %.4f : %.4f +/- %.4f" % (i, p25, p50, p75, p50, 0.741 * (p75-p25))
