import umbridge 
import tinyDA as tda
import numpy as np
from scipy.integrate import solve_ivp
import scipy.stats as stats
import arviz as az



# Inelegant repetition but necessary
class PredatorPreyModel:
    def __init__(self, datapoints):
        
        # set the timesteps, where we are collecting the model output
        self.datapoints = datapoints
        
        # set the span of the integration.
        self.t_span = [0, self.datapoints[-1]]
        
    def dydx(self, t, y, a, b, c, d):
        # Lotka-Volterra Model model, see e.g. https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
        return np.array([a*y[0] - b*y[0]*y[1], c*y[0]*y[1] - d*y[1]])

    def __call__(self, parameters):
        
        # extract the parameters, and take the exponential to keep them positive
        P_0, Q_0, a, b, c, d = np.exp(parameters)
        
        # solve the initial value problem.
        self.y = solve_ivp(lambda t, y: self.dydx(t, y, a, b, c, d), self.t_span, np.array([P_0, Q_0]), t_eval=self.datapoints) 
        
        # return the results, only if the integration succeeded.
        if self.y.success:
            return self.y.y.flatten(), True
        else:
            return np.nan, False


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


# set the integration points.
t_span = [0,12]
n_eval = 1000
t_eval = np.linspace(t_span[0], t_span[1], n_eval)


# fine model
n_data_l2 = 25 # number of datapoints
t_eval_l2 = np.linspace(t_span[0], t_span[1], n_data_l2) # datapoints
my_model_l2 = PredatorPreyModel(t_eval_l2) # initialise model


# coarse model (stops integrating early)
n_data_l1 = 16 # number of datapoints
t_eval_l1 = t_eval_l2[:n_data_l1] # datapoints (the first 16 of the fine model)
my_model_l1 = PredatorPreyModel(t_eval_l1) # initialise model


# coarse model (stops integrating even earlier)
n_data_l0 = 8 # number of datapoints
t_eval_l0 = t_eval_l2[:n_data_l0] # datapoints (the first 8 of the fine model)
my_model_l0 = PredatorPreyModel(t_eval_l0) # initialise model


# Collect "real" data from numerical model
noise_l2 = np.random.normal(scale=sigma, size=(t_eval_l2.size,2)) # fine noise
data_l2 = my_model_l2(true_parameters)[0] + np.hstack((noise_l2[:,0], noise_l2[:,1])) # noisy fine data.
data_l2[data_l2 < 0] = 0 # make sure all the data is positive.

noise_l1 = np.hstack((noise_l2[:n_data_l1,0], noise_l2[:n_data_l1,1])) # coarse noise
data_l1 = my_model_l1(true_parameters)[0] + noise_l1 # noisy coarse data
data_l1[data_l1 < 0] = 0 # make sure all the data is positive.

noise_l0 = np.hstack((noise_l2[:n_data_l0,0], noise_l2[:n_data_l0,1])) # coarse noise
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

# definea umbridge models, overiding exact model
my_model_l0 = tda.UmBridgeModel(umbridge.HTTPModel("http://localhost:4242", "l0"))
my_model_l1 = tda.UmBridgeModel(umbridge.HTTPModel("http://localhost:4242", "l1"))
my_model_l2 = tda.UmBridgeModel(umbridge.HTTPModel("http://localhost:4242", "l2"))

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
iterations = 2
burnin = 1
my_chain = tda.sample(my_posteriors, my_proposal, iterations=iterations, n_chains=5, subchain_length=[5, 10])

idata = tda.to_inference_data(my_chain, level=2, burnin=burnin)
print(az.summary(idata))
