import umbridge 
import tinyDA as tda
import numpy as np
import scipy.stats as stats
import arviz as az


# define models
my_model_l0 = tda.UmBridgeModel(umbridge.HTTPModel("http://localhost:4249", "l0"))
my_model_l1 = tda.UmBridgeModel(umbridge.HTTPModel("http://localhost:4249", "l1"))
my_model_l2 = tda.UmBridgeModel(umbridge.HTTPModel("http://localhost:4249", "l2"))

# #set the true parameters
P_0 = 10
Q_0 = 5
a = 1.0
b = 0.3
c = 0.2
d = 1.0

# collect the parameters in a vector and take the logarithm. 
# we sample the log of the parameters and take the exponential
# inside the model to keep the parameters positive.
true_parameters = np.log(np.array([P_0, Q_0, a, b, c, d]))


# set the noise level
sigma = 1.0

noise_l2 = np.random.normal(scale=sigma, size=(25,2)) # fine noise
data_l2 = my_model_l2(true_parameters)[0] + np.hstack((noise_l2[:,0], noise_l2[:,1])) # noisy fine data.
data_l2[data_l2 < 0] = 0 # make sure all the data is positive.

noise_l1 = np.hstack((noise_l2[:16,0], noise_l2[:16,1])) # coarse noise
data_l1 = my_model_l1(true_parameters)[0] + noise_l1 # noisy coarse data
data_l1[data_l1 < 0] = 0 # make sure all the data is positive.

noise_l0 = np.hstack((noise_l2[:8,0], noise_l2[:8,1])) # coarse noise
data_l0 = my_model_l0(true_parameters)[0] + noise_l0 # noisy coarse data
data_l0[data_l0 < 0] = 0 # make sure all the data is positive



# prior distribution
mean_prior = np.array([np.log(data_l2[0]), np.log(data_l2[25]), 0, -1, -1.5, 0])
cov_prior = np.diag([0.1, 0.1, 0.001, 0.1, 0.1, 0.001])
my_prior = stats.multivariate_normal(mean_prior, cov_prior)


# define the likelihood
cov_likelihood_l2 = sigma**2*np.eye(data_l2.size)
cov_likelihood_l1 = sigma**2*np.eye(data_l1.size)
cov_likelihood_l0 = sigma**2*np.eye(data_l0.size)

my_loglike_l2 = tda.GaussianLogLike(data_l2, cov_likelihood_l2)
my_loglike_l1 = tda.GaussianLogLike(data_l1, cov_likelihood_l1)
my_loglike_l0 = tda.GaussianLogLike(data_l0, cov_likelihood_l0)


# set up the link factories
my_posterior_l2 = tda.Posterior(my_prior, my_loglike_l2, my_model_l2)
my_posterior_l1 = tda.Posterior(my_prior, my_loglike_l1, my_model_l1)
my_posterior_l0 = tda.Posterior(my_prior, my_loglike_l0, my_model_l0)

my_posteriors = [my_posterior_l0, my_posterior_l1, my_posterior_l2]


# random walk Metropolis
rwmh_cov = np.eye(6)
rmwh_scaling = 0.1
rwmh_adaptive = True
my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive)



# initialise the chain
iterations = 10
burnin = 5
my_chain = tda.sample(my_posteriors, my_proposal, iterations=iterations, n_chains=2, subchain_length=2, force_sequential=True)

idata = tda.to_inference_data(my_chain, level=2, burnin=burnin)
az.summary(idata)
