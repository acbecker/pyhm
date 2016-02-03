import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import emcee
import priors
import posteriors
import summary

def pooled(treatN, treatD):

    description = """ In this problem, a treatment is applied to
    treatN patients, from 5 randomly selected centers participating in
    a treatment study.  In each center, treatD deaths are recorded.
    The first analysis is to calculate the shared survival
    probability, ignoring any differences in center.  This is
    considered a pooled analysis.

    The model has a shared probability of death of p, which is
    represented as a binomial distribution in treatN take p*treatN.
    We use a uniform prior on p.

    Measurement Level Model:
    (d_i | n_i p_i) ~ binomial(n_i, p)

    Population Level Model; or Prior Level:
    Define prior(p)
    
    """
    
    priorP = priors.UniformPrior(0.0, 1.0)
    def lnprior(params):
        p = params
        return priorP.lnlike(p)

    def lnlike(params, *args):
        p = params
        lp = lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        nn, dd = args
        bin = [stats.binom.logpmf(d, n, p) for n,d in zip(nn,dd)]
        return np.sum(bin) + lp

    pGuess = np.mean(1.0*treatD/treatN)
    ndim, nwalkers, nburn, nstep = 1, 100, 100, 1000
    pos = [np.array((pGuess)) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(treatN, treatD))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    import pdb; pdb.set_trace()

def randomEffects(treatN, treatD):

    description = """In this case, we assume that the probability p is
    different in each trial, but that they are drawn from a common
    distribution.  This distribution is a beta(a, b), where a and b
    are hyperparameters of the model.  We will apply common priors on
    a and b of an exponential distribution with scale = 0.

    Measurement Level Model:
    (d_i | n_i p_i) ~ binomial(n_i, p_i)

    Population Level Model; or Prior Level:
    (p_i | a, b) ~ beta(a, b)

    Hyperprior Level:
    Define prior(a, b)
    
    """

    priorA = priors.ExponentialPrior(100.)
    priorB = priors.ExponentialPrior(100.)
    def lnprior(params):
        a = params[-2]
        b = params[-1]
        return priorA.lnlike(a) + priorB.lnlike(b)

    def lnlike(params, *args):
        # Hyperprior
        lp = lnprior(params)
        if not np.isfinite(lp):
            return -np.inf

        # Likelihood of p_i under the population level model
        a = params[-2]
        b = params[-1]
        priorP = stats.beta(a, b)
        lnpriorP = np.sum(priorP.logpdf(params[:-2]))
        if not np.isfinite(lnpriorP):
            return -np.inf

        # Likelihood of d_i under the measurement level model
        nn, dd = args
        lnlikeP = np.sum([stats.binom.logpmf(d, n, p) for d,n,p in zip(dd,nn,params[:-2])])

        return lp + lnpriorP + lnlikeP

    pGuess = np.mean(1.0*treatD/treatN)
    aGuess = 3.0
    bGuess = 191.0
    ndim, nwalkers, nburn, nstep = len(treatN)+2, 100, 1000, 10000
    guess = [pGuess,]*len(treatN)
    guess.append(aGuess)
    guess.append(bGuess)
    pos = [np.array(guess) + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(treatN, treatD))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    import pdb; pdb.set_trace()
    summ = summary.Summary()
    summ.summarize(sampler.chain)

if __name__ == "__main__":
    #treatN = np.array((86, 69, 71, 113, 103))
    #treatD = np.array((2,  2,  1,  1,   1))
    #pooled(treatN, treatD)

    treatN = np.array((86, 69, 71, 113, 103, 200))
    treatD = np.array((2,  2,  1,  1,   1,   1))
    randomEffects(treatN, treatD)
