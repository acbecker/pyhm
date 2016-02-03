import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

description = """ This one is a bit different, and explains how to use
   odds ratios, and the beta distribution, to understand the difference
   between a case and control sample

               Cancer   Control
   Smokers     83       72
   Nonsmokers  3        14

   Here we have 86 individuals in both case and control samples.  83
   smokers obtained cancer, while 3 nonsmokers obtained cancer.  72
   smokers did not obtain cancer, while 3 nonsmokers did not obtain
   cancer.  Ask the question: is there a significant difference
   between the smoking habits in the two groups?

   pL: population proportion of lung-cancer patients who smoke = 83/86
   pC: population proportion of control who smoke  = 72/86

   The question may be be answered by examining the posterior
   distribution of pL - pC.  Or the odds ratio

      pL/(1-pL)  /  pC/(1-pC)

   The log odds ratio lambda is less skewed, so ask the question
   p(lambda>0 | data)

   Overall likelihood is :

     L(pL, pC) = pL^83(1 - pL)^3 pC^72(1 - pC)^14

   pL has a Beta distribution, specifically Beta(83,3)
   pC has Beta(72,14)

   Draw 1000 pL' from scipy.stats.beta(83, 3)
   Draw 1000 pC' from scipy.stats.beta(72, 14)
   Calculate the number of times the odds ratio is > 0!
   """

betaL   = stats.beta(83, 3)
betaC   = stats.beta(72, 14)

ndraws  = 10000
pLp     = betaL.rvs(size=ndraws)
pCp     = betaC.rvs(size=ndraws)

numer   = pLp / (1 - pLp)
denom   = pCp / (1 - pCp)
logodds = np.log(numer/denom)

print "Expectation value of logodds  : %.4f" % (np.mean(logodds))
print "Standard deviation of logodds : %.4f" % (np.std(logodds))
print "P(logodds > 0 | data)         : %.4f" % (1.0*len(np.where(logodds>0)[0])*1.0/ndraws)
