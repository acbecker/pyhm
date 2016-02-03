import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import emcee
import parameters
import summary

def pooled(df, outcomeKey, predictorKeys, poolByKey):
    description = """
    Measurement Level Model:
      (y_i | alpha, beta, sigma_i) ~ normal(alpha + beta * values_i, sigma_i**2)

    Population Level Model; or Prior Level:
    
      No priors
    """
    y  = np.log(df[outcomeKey]).values.ravel()
    y[~np.isfinite(y)] = -1
    x  = df[predictorKeys].values.ravel()
    p  = df[poolByKey].values.ravel()

    poolValues = df[poolByKey].unique()
    npar = len(poolValues)
    pidx = []
    dy   = np.empty(npar)
    for i, poolValue in enumerate(poolValues):
        idx = (p == poolValue)
        pidx.append(idx)
        dy[i] = np.std(y[idx])
    dy[dy==0.0] = y[dy==0.0]
    
    def lnlike(params, *args):
        a = params[:-1]
        b = params[-1]
        xx, yy, dyy, ppidx = args
        lnl = 0
        for i, idx in enumerate(ppidx):
            lnl += np.sum([stats.norm.logpdf(yyy, a[i] + b * xxx, dy[i]) for yyy, xxx in zip(yy[idx], xx[idx])])
        if not np.isfinite(lnl):
            return -np.inf
        return lnl

    guess = npar * [0.9,] + [-0.61,]
    ndim, nwalkers, nburn, nstep = len(guess), 2*len(guess), 10, 100
    pos = [np.array(guess) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y, dy, pidx))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    summ = summary.Summary()
    summ.summarize(sampler.chain)
    import pdb; pdb.set_trace()
    

def unpooled(df, outcomeKey, predictorKeys):
    description = """
    Measurement Level Model:

      (y_i | alpha_ji, beta, sigma_i) ~ normal(alpha_ji + beta * values_i, sigma_i**2)

    where j is the county from which the measurement was taken.

    Population Level Model; or Prior Level:
    
      No priors
    """
    y  = np.log(df[outcomeKey]).values.ravel()
    y[~np.isfinite(y)] = -1
    dy = np.std(y)
    x  = df[predictorKeys].values.ravel()
    import pdb;pdb.set_trace()
    def lnlike(params, *args):
        a = params[0]
        b = params[1]
        xx, yy, dyy = args
        lnlikes = [stats.norm.logpdf(yyy, a + b * xxx, dyy) for yyy, xxx in zip(yy, xx)]
        lnl = np.sum(lnlikes)
        if not np.isfinite(lnl):
            return -np.inf
        return lnl

    aGuess, bGuess = 1.3, -0.61
    ndim, nwalkers, nburn, nstep = 2, 2*2, 100, 1000
    pos = [np.array((aGuess, bGuess)) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y, dy))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    summ = summary.Summary()
    summ.summarize(sampler.chain)
    
    
def multilevel(values, sigma):
    description = """
    Measurement Level Model:

      (y_i | alpha_ji, beta, sigma) ~ normal(alpha_ji + beta * values_i, sigma**2)

    Population Level Model; or Prior Level:

      (alpha_ji | mu_a, alpha_a) = normal(mu_a, alpha_a**2)

    Hyperprior Level:

      Define prior(mu_a, alpha_a, sigma, beta)

    """
    y  = np.log(df[outcomeKey]).values.ravel()
    y[~np.isfinite(y)] = -1
    x  = df[predictorKeys].values.ravel()
    p  = df[poolByKey].values.ravel()

    poolValues = df[poolByKey].unique()
    npar = len(poolValues)
    pidx = []
    dy   = np.empty(npar)
    for i, poolValue in enumerate(poolValues):
        idx = (p == poolValue)
        pidx.append(idx)
        dy[i] = np.std(y[idx])
    dy[dy==0.0] = y[dy==0.0]


    priorMu_a = priors.Uniform(0, 100)
    priorSigma_a = priors.Uniform(0, 100)

    def lnprior(params):
        mu = params[-2]
        sigma = params[-1]
        return priorMu_a.lnlike(mu) + priorSigma_a.lnlike(sigma)

    def lnlike(params, *args):
        xx, yy, dyy, ppidx = args

        alpha = params[:-3]
        b     = params[-3]
        mu    = params[-2]
        sigma = params[-1]

        # Hyperprior
        lnpriorH = lnprior(params)
        if not np.isfinite(lnpriorH):
            return -np.inf

        # Likelihood of a_i under the population level model
        lnpriorP = np.sum([stats.norm.logpdf(a, mu, sigma) for a in alpha])
        if not np.isfinite(lnpriorP):
            return -np.inf

        # Likelihood of y_i under the measurement level model
        lnlikeM = 0
        for i, idx in enumerate(ppidx):
            lnlikeM += np.sum([stats.norm.logpdf(yyy, alpha[i] + b * xxx, dy[i]) for yyy, xxx in zip(yy[idx], xx[idx])])
        if not np.isfinite(lnlikeM):
            return -np.inf
        return lnpriorH + lnpriorP + lnlikeM

    guess = npar * [0.9,] + [-0.61,]
    ndim, nwalkers, nburn, nstep = len(guess), 2*len(guess), 10, 100
    pos = [np.array(guess) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y, dy, pidx))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    summ = summary.Summary()
    summ.summarize(sampler.chain)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    infile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/arm/radon/srrs2.dat")
    df     = pd.read_csv(infile, "df", skipinitialspace=True, delimiter=",")
    df     = df[df.state == "MN"]
    outcomeKey    = "activity"
    predictorKeys = ["floor",]
    poolByKey     = "cntyfips"
    #unpooled(df, outcomeKey, predictorKeys)
    pooled(df, outcomeKey, predictorKeys, poolByKey)
    
