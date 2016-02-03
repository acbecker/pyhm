import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import emcee
import priors
import posteriors
import os

def pooled(estimates, sds):
    description = """
    Measurement Level Model:
    (y_i | theta_i, sigma_i) ~ normal(theta, sigma_i)

    Population Level Model; or Prior Level:
    Define prior(theta)
    """
    
    def lnlike(params, *args):
        mu = params
        likes = [stats.norm.logpdf(e, mu, s) for e,s in zip(estimates, sds)]
        return np.sum(likes)

    muGuess = np.mean(estimates)
    ndim, nwalkers, nburn, nstep = 1, 100, 100, 1000
    pos = [np.array((muGuess,)) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(estimates, sds))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    import pdb; pdb.set_trace()

def hierarchical(estimates, sds):
    description = """
    Measurement Level Model:
    (y_i | theta_i, sigma_i) ~ normal(theta_i, sigma_i)

    Population Level Model; or Prior Level:
    (theta_i | mu tau) ~ normal(mu, tau)

    Hyperprior Level:
    Define prior(mu, tau)
    """

    priorMu = priors.Uniform(0, 100)
    priorTheta = priors.Uniform(0, 100)
    def lnprior(params):
        mu = params[-2]
        tau = params[-1]
        return priorMu.lnlike(mu) + priorTheta.lnlike(theta)
    
    def lnlike(params, *args):
        thetas = params[:-2]
        mu = params[-2]
        tau = params[-1]

        # Hyperprior
        lnpriorH = lnprior(params)
        if not np.isfinite(lnpriorH):
            return -np.inf

        # Likelihood of theta_i under the population level model
        lnpriorT = np.sum([stats.norm.logpdf(t, mu, tau) for t in thetas])
        if not np.isfinite(lnpriorT):
            return -np.inf

        # Likelihood of y_i under the measurement level model
        lnlikeY = np.sum([stats.norm.logpdf(e, t, s) for e,t,s in zip(estimates, thetas, sds)])
        return lnpriorH + lnpriorT + lnlikeY

    guess = [np.mean(estimates),]*len(estimates)
    guess.append(np.mean(estimates))
    guess.append(np.std(estimates))
    ndim, nwalkers, nburn, nstep = len(estimates)+2, 100, 1000, 10000
    pos = [np.array(guess) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(estimates, sds))
    pos, prob, state = sampler.run_mcmc(pos, nburn)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep, rstate0=state)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    datfile = os.path.join(os.path.dirname(__file__), "eightSchools.dat")
    estimates, sds = np.loadtxt(datfile, unpack=True, comments="#", usecols=[1,2])
    #pooled(estimates, sds)
    hierarchical(estimates, sds)
