import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as opt
import emcee
import triangle, walkers
import priors

np.random.seed(666)

def randomData(a, b, sig, npts = 100):
    x    = np.random.random(npts)
    mean = a + b * x
    y = stats.norm(loc=mean, scale=sig).rvs(size=mean.shape)
    return x, y

if __name__ == "__main__":
    aTrue    = 7.1
    bTrue    = 10.0
    sigTrue  = np.sqrt(2.0)
    aGuess   = 6.9
    bGuess   = 10.5
    sigGuess = np.sqrt(1.8)
    
    x, y = randomData(aTrue, bTrue, sigTrue)
    yerrTrue = sigTrue*np.ones_like(x)

    sigPrior = priors.HalfCauchyPrior(0, 0.1)
    # Step 5: MCMC modeling, unknown errors, prior on sigma
    def chi2(params, *args):
        a, b, sigma = params
        xf, yf = args
        prediction = a + b * xf
        chi2  = np.sum( (prediction - yf)**2 / sigma**2 )
        chi2 += np.log(2 * np.pi * sigma**2)
        return chi2
    def lnprior(params):
        a, b, sigma = params
        #if sigma < 1e-10 or sigma > 10:
        #    return -np.inf
        #return 0.0
        return sigPrior.lnlike(sigma)
    def lnlike(params, *args):
        lp = lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return -0.5 * chi2(params, *args) + lp

    result = opt.minimize(chi2, [aGuess, bGuess, sigGuess], args=(x, y), method="BFGS")
    print "Nonlinear fit, no errors"
    print " a'   = %.3f +/- %.3f" % (result.x[0], np.sqrt(result.hess_inv[0][0]))
    print " b'   = %.3f +/- %.3f" % (result.x[1], np.sqrt(result.hess_inv[1][1]))
    print " sig' = %.3f +/- %.3f" % (result.x[2], np.sqrt(result.hess_inv[2][2]))

    ndim, nwalkers, nburn, nstep = 3, 100, 1000, 10000
    pos = [np.array((aGuess, bGuess, sigGuess)) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    fig = plt.figure()
    sp = fig.add_subplot(111)
    sp.errorbar(x, y, yerr=yerrTrue, fmt="ro")
    sp.plot(x, aTrue + bTrue * x, "r-")
    flatchain = sampler.flatchain
    print "MCMC analysis"
    flata   = flatchain[:,0]
    flatb   = flatchain[:,1]
    flatsig = flatchain[:,2]
    print " a = %.3f +%.3f -%.3f" % (np.percentile(flata, 50),
                                     np.percentile(flata, 50)-np.percentile(flata, 50-68.27/2),
                                     np.percentile(flata, 50+68.27/2)-np.percentile(flata, 50),
                                     )
    print " b = %.3f +%.3f -%.3f" % (np.percentile(flatb, 50),
                                     np.percentile(flatb, 50)-np.percentile(flatb, 50-68.27/2),
                                     np.percentile(flatb, 50+68.27/2)-np.percentile(flatb, 50),
                                     )
    print " sig = %.3f +%.3f -%.3f" % (np.percentile(flatsig, 50),
                                       np.percentile(flatsig, 50)-np.percentile(flatsig, 50-68.27/2),
                                       np.percentile(flatsig, 50+68.27/2)-np.percentile(flatsig, 50),
                                       )

    for a, b in flatchain[np.random.randint(len(flatchain), size=100)][:,:2]:
        sp.plot(x, a + b*x, color="k", alpha=0.05)
    triangle.triangle(flatchain, ("a", "b", "sig"))
    walkers.walkers(sampler.chain, ("a", "b", "sig"))
    plt.show()    
