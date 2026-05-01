import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal,uniform
from scipy.spatial import distance_matrix

import tinyDA as tda
import umbridge

np.random.seed(111)

# 2D uniform dist
class uniform_2D:
    def __init__(self, locx=0, locy=0, scalex=1, scaley=1):
        self.loc = [locx, locy]
        self.scale = [scalex, scaley]
        self.distx = uniform(locx, scalex)
        self.disty = uniform(locy, scaley)
            
    def rvs(self, x=1):
        samplex = self.distx.rvs(x)
        sampley = self.disty.rvs(x)
        return np.hstack([samplex, sampley])

    def pdf(self, x):
        return self.distx.pdf(x[0]) * self.disty.pdf(x[1])

    def logpdf(self, x):
        return self.distx.logpdf(x[0]) + self.disty.logpdf(x[1])


# connect to the UM-Bridge model.
while True:
    try:
        umbridge_model = umbridge.HTTPModel('http://localhost:4242', "forward")
        break
    except Exception:
        time.sleep(15)

# wrap the UM-Bridge model in the tinyDA UM-Bridge interface.
my_model0 = tda.UmBridgeModel(umbridge_model, umbridge_config={"level": 0})
my_model1 = tda.UmBridgeModel(umbridge_model, umbridge_config={"level": 1})
my_model2 = tda.UmBridgeModel(umbridge_model, umbridge_config={"level": 2})

# set up the prior.
my_prior = uniform_2D(-200000, -200000, 400000, 400000)
# my_prior = multivariate_normal([0.0, 0.0], np.diag([1.0, 1.0]) * 10000)


# set the likelihood

# data_l0 = np.array([1803.779, 5216.222, 2.342, 0.6030])
data = np.array([1813.92 / 60.0, 5278.92 / 60.0, 1.907, 0.6368])

cov_likelihood_l0 = np.diag([2.5, 2.5, 0.15, 0.15])
cov_likelihood_l1 = 0.5 * np.diag([2.5, 2.5, 0.15, 0.15])
cov_likelihood_l2 = 0.1 * np.diag([2.5, 2.5, 0.15, 0.15])

# L2 probe18: 1803.799, 2.342 Probe19: 5216.222, 0.6030
# Real probe18: 1813.92, 1.907 Probe19: 5278.92, 0.6368
my_loglike_l0 = tda.GaussianLogLike(data, cov_likelihood_l0)
my_loglike_l1 = tda.GaussianLogLike(data, cov_likelihood_l1)
my_loglike_l2 = tda.GaussianLogLike(data, cov_likelihood_l2)

# initialise the LinkFactory
# the umbridge model contains two levels 

my_posterior_l0 = tda.Posterior(my_prior, my_loglike_l0, my_model0)
my_posterior_l1 = tda.Posterior(my_prior, my_loglike_l1, my_model1)
my_posterior_l2 = tda.Posterior(my_prior, my_loglike_l2, my_model2)

my_posteriors = [my_posterior_l0, my_posterior_l1, my_posterior_l2] 


# Various choices of proposal
# preconditioned Crank-Nicolson
# pcn_scaling = 0.1
# pcn_adaptive = True
# my_proposal = tda.CrankNicolson(scaling=pcn_scaling, adaptive=pcn_adaptive)

# random walk Metropolis
rwmh_cov = np.diag([4000, 4000])
rmwh_scaling = 1.0
rwmh_adaptive = True
my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive)

# Adaptive Metropolis
#am_cov = np.eye(true_parameters.size)
#am_t0 = 100
#am_sd = None
#am_epsilon = 1e-6
#am_adaptive = True
#my_proposal = tda.AdaptiveMetropolis(C0=am_cov, t0=am_t0, sd=am_sd, epsilon=am_epsilon)

# dream_m0 = 1000
# dream_delta = 1
# dream_Z_method = 'lhs'
# dream_adaptive = True
# my_proposal = tda.DREAMZ(M0=dream_m0, delta=dream_delta, Z_method=dream_Z_method, adaptive=dream_adaptive)

iterations = 20
burnin = 5

# Initialise chain
my_chains = tda.sample(my_posteriors, my_proposal, subchain_length=[50, 5], iterations=iterations, n_chains=10, force_sequential=False)


import arviz as az
# convert the tinyDA chains to an ArViz InferenceData object.
idata = tda.to_inference_data(my_chains, burnin=burnin)
idata.to_netcdf("tinyda.nc")

# display posterior summary statistics.
print(az.summary(idata))

# plot posterior kernel densities and traces.
az.plot_trace(idata)
plt.savefig("MCMC_trace.png")

# extract the parameters from the chains.
parameters = [link.parameters for link in my_chains['chain_0'][burnin:] + my_chains['chain_1'][burnin:]]



