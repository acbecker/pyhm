import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as opt
import emcee
import triangle, walkers

np.random.seed(666)

def randomData(a, b, var, npts = 100):
    x    = np.random.random(npts)
    mean = a + b * x

    # NOTE: IT IS EXTREMELY IMPORTANT TO HAVE SIZE IN THE RVS CALL
    # OTHERWISE ONLY 1 RANDOM NUMBER IS DRAWN AND APPLIED TO ALL DATA.
    # E.g.
    # >>> loc = np.array([1, 10, 100])
    # >>> stats.norm(loc=loc, scale=2.0).rvs()
    # array([   2.27343007,   11.27343007,  101.27343007])
    # >>> stats.norm(loc=loc, scale=2.0).rvs(size=loc.shape)
    # array([ -2.55375614,  12.15124866,  98.24302567])
    y = stats.norm(loc=mean, scale=np.sqrt(var)).rvs(size=mean.shape)
    
    return x, y

if __name__ == "__main__":
    aTrue   = 7.1
    bTrue   = 10.0
    varTrue = 2.0
    varFalse = 1.0

    aGuess = 6.9
    bGuess = 10.5
    varGuess = 1.8
    
    x, y = randomData(aTrue, bTrue, varTrue)
    yerrTrue = np.sqrt(varTrue)*np.ones_like(x)

    # Step 1: Least squares fitting
    # Notation from http://arxiv.org/abs/1008.4686
    #   and http://dan.iel.fm/emcee/current/user/line/
    A   = np.vstack((np.ones_like(x), x)).T
    C   = np.diag(varTrue*np.ones_like(x))
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    a_ls, b_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    print "Linear fit"
    print " a = %.3f +/- %.3f" % (a_ls, np.sqrt(cov[0][0]))
    print " b = %.3f +/- %.3f" % (b_ls, np.sqrt(cov[1][1]))
    fig = plt.figure()
    sp = fig.add_subplot(111)
    sp.errorbar(x, y, yerr=yerrTrue, fmt="ro")
    sp.plot(x, a_ls + b_ls * x, "k-")
    sp.plot(x, aTrue + bTrue * x, "r-")

    # Step 2a: Nonlinear fitting with correct error bars
    def chi2(params, *args):
        a, b = params
        xf, yf, dyf = args
        prediction = a + b * xf
        chi2 = np.sum( ((prediction - yf) / dyf)**2)
        return chi2
    def lnlike(params, *args):
        return -0.5 * chi2(params, *args)

    result = opt.minimize(chi2, [aGuess, bGuess], args=(x, y, yerrTrue), method="BFGS")
    print "Nonlinear fit, correct errors"
    print " a = %.3f +/- %.3f" % (result.x[0], np.sqrt(result.hess_inv[0][0]))
    print " b = %.3f +/- %.3f" % (result.x[1], np.sqrt(result.hess_inv[1][1]))
    # Step 2b: Nonlinear fitting with incorrect error bars
    yerrFalse = np.sqrt(varFalse)*np.ones_like(x)
    result = opt.minimize(chi2, [aGuess, bGuess], args=(x, y, yerrFalse), method="BFGS")
    print "Nonlinear fit, incorrect errors, chi2 = %.3f" % (result.fun)
    print " a = %.3f +/- %.3f" % (result.x[0], np.sqrt(result.hess_inv[0][0]))
    print " b = %.3f +/- %.3f" % (result.x[1], np.sqrt(result.hess_inv[1][1]))
    # Attempt to correct error bars using chi2; chi2/dof should be 1
    chi2dof = result.fun / (len(x) - 2)
    # Step 2c: Nonlinear fitting with corrected
    yerrCorr = yerrFalse * np.sqrt(chi2dof)
    result = opt.minimize(chi2, [aGuess, bGuess], args=(x, y, yerrCorr), method="BFGS")
    print "Nonlinear fit, corrected errors, chi2 = %.3f" % (result.fun)
    print " a' = %.3f +/- %.3f" % (result.x[0], np.sqrt(result.hess_inv[0][0]))
    print " b' = %.3f +/- %.3f" % (result.x[1], np.sqrt(result.hess_inv[1][1]))

    # Step 3: MCMC modeling, assuming the errors are correct
    ndim, nwalkers, nburn, nstep = 2, 100, 1000, 10000
    pos = [result.x + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y, np.sqrt(varTrue)*np.ones_like(x)))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    fig = plt.figure()
    sp = fig.add_subplot(111)
    sp.errorbar(x, y, yerr=yerrTrue, fmt="ro")
    sp.plot(x, aTrue + bTrue * x, "r-")
    flatchain = sampler.flatchain
    print "MCMC analysis 1"
    print " a = %.3f +%.3f -%.3f" % (np.percentile(flatchain[:,0], 50),
                                     np.percentile(flatchain[:,0], 50)-np.percentile(flatchain[:,0], 50-68.27/2),
                                     np.percentile(flatchain[:,0], 50+68.27/2)-np.percentile(flatchain[:,0], 50),
                                     )
    print " b = %.3f +%.3f -%.3f" % (np.percentile(flatchain[:,1], 50),
                                     np.percentile(flatchain[:,1], 50)-np.percentile(flatchain[:,1], 50-68.27/2),
                                     np.percentile(flatchain[:,1], 50+68.27/2)-np.percentile(flatchain[:,1], 50),
                                     )
    
    for a, b in flatchain[np.random.randint(len(flatchain), size=100)]:
        sp.plot(x, a + b*x, color="k", alpha=0.05)
    # Plot the joint distributions
    triangle.triangle(flatchain, ("a", "b"))
    
    # Step 4: MCMC modeling, unknown errors
    def chi2(params, *args):
        a, b, lnvar = params
        xf, yf = args
        prediction = a + b * xf
        chi2  = np.sum( (prediction - yf)**2 / np.exp(lnvar) )
        chi2 += np.log(2 * np.pi) + lnvar
        return chi2
    def lnprior(params):
        a, b, lnvar = params
        if -10.0 < lnvar < 1.0:
            return 0.0
        return -np.inf
    def lnlike(params, *args):
        lp = lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return -0.5 * chi2(params, *args)

    result = opt.minimize(chi2, [aGuess, bGuess, varGuess], args=(x, y), method="BFGS")
    print "Nonlinear fit, no errors"
    print " a'   = %.3f +/- %.3f" % (result.x[0], np.sqrt(result.hess_inv[0][0]))
    print " b'   = %.3f +/- %.3f" % (result.x[1], np.sqrt(result.hess_inv[1][1]))
    print " var' = %.3f +/- %.3f" % (result.x[2], np.sqrt(result.hess_inv[2][2]))

    ndim, nwalkers, nburn, nstep = 3, 100, 1000, 10000
    pos = [np.array((aGuess, bGuess, np.log(varGuess))) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    fig = plt.figure()
    sp = fig.add_subplot(111)
    sp.errorbar(x, y, yerr=yerrTrue, fmt="ro")
    sp.plot(x, aTrue + bTrue * x, "r-")
    flatchain = sampler.flatchain
    print "MCMC analysis 2"
    flata   = flatchain[:,0]
    flatb   = flatchain[:,1]
    flatvar = np.exp(flatchain[:,2])
    print " a = %.3f +%.3f -%.3f" % (np.percentile(flata, 50),
                                     np.percentile(flata, 50)-np.percentile(flata, 50-68.27/2),
                                     np.percentile(flata, 50+68.27/2)-np.percentile(flata, 50),
                                     )
    print " b = %.3f +%.3f -%.3f" % (np.percentile(flatb, 50),
                                     np.percentile(flatb, 50)-np.percentile(flatb, 50-68.27/2),
                                     np.percentile(flatb, 50+68.27/2)-np.percentile(flatb, 50),
                                     )
    print " var = %.3f +%.3f -%.3f" % (np.percentile(flatvar, 50),
                                       np.percentile(flatvar, 50)-np.percentile(flatvar, 50-68.27/2),
                                       np.percentile(flatvar, 50+68.27/2)-np.percentile(flatvar, 50),
                                       )

    for a, b in flatchain[np.random.randint(len(flatchain), size=100)][:,:2]:
        sp.plot(x, a + b*x, color="k", alpha=0.05)
    triangle.triangle(flatchain, ("a", "b", "var"))
    walkers.walkers(sampler.chain, ("a", "b", "var"))
    plt.show()

