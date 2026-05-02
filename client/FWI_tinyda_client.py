import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal,uniform
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter1d
import scipy
import pandas as pd
import tinyDA as tda
import umbridge
import torch
from GP.weighted_gp import WeightedGP

np.random.seed(111)

# Wasserstein likelihoods
# Linear scaling
class WassersteinLoglikeLinear:
    def __init__(self, data, shift=0.0, lam=1.0):
        self.lam = lam
        self.shift = shift
        self.data = np.asarray(data, dtype=float)

    def _normalize(self, x, shift):
        shifted = x + np.abs(shift)
        return shifted / len(x), shift

    def _row_loglike(self, x_row, data_row, shift):
        xn, _ = self._normalize(x_row, shift)
        datan, _ = self._normalize(data_row, shift)
        return -self.lam * np.sum((np.sort(xn) - np.sort(datan)) ** 2)

    def loglike(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 100)
        return self._row_loglike(x[0], self.data[0], self.shift[0]) + self._row_loglike(x[1], self.data[1], self.shift[1])


# Engquist/Yang scaling
# This is specific to two probes.
# We calculate the Wasserstein distance separately and add

class WassersteinLoglike:
    def __init__(self, data, lam=1.0):
        self.lam = lam
        self.data = np.asarray(data, dtype=float)

    def _normalize(self, x, data_row):
        c = min(data_row.min(), x.min())
        pos = x >= 0
        raw = np.where(pos, x + 1.0 / c, (1.0 / c) * np.exp(c * x))
        return raw / raw.sum()

    def _row_loglike(self, x_row, data_row):
        xn    = self._normalize(x_row,  data_row)
        datan = self._normalize(data_row, data_row)
        W1 = np.sum((np.sort(xn) - np.sort( datan)) ** 2)
        W2 = np.sum((np.sort(-xn) - np.sort(-datan)) ** 2)
        return -self.lam * (W1 + W2)

    def loglike(self, x):
        x = np.asarray(x, dtype=float).reshape(2, 100)

        return self._row_loglike(x[0], self.data[0]) \
             + self._row_loglike(x[1], self.data[1])
   
# 2D uniform dist for initial displacement
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
        return (self.distx.pdf(x[0]) * self.disty.pdf(x[1]))

    def logpdf(self, x):
        return self.distx.logpdf(x[0]) + self.disty.logpdf(x[1])
        
        
def lowpass(data: np.ndarray, cutoff, sample_rate, poles=5):
    """
    Low pass filter to remove high frequency noise in the data
    """
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def processBuoyData(buoy, cutoff=0.0035):
    processed = pd.DataFrame()
    processed["t"]= pd.to_numeric(59 + buoy[2]+ buoy[3]/float(24)+ buoy[4]/float(24*60) + buoy[5]/float(24*3600) -70.2422) *24 * 60 * 60
    processed["ssha"]= pd.to_numeric(buoy[8])

    processed = processed[processed["t"].between(0.0, 6000.0, inclusive="both")]
    
    sample_rate = len(processed["t"]) / (max(processed["t"]) - min(processed["t"]))
    processed["ssha"] = lowpass(processed["ssha"].values, cutoff, sample_rate)
    
    return processed

# Loads GP to use saved scaler
with open("../GP/Series18.pkl", "rb") as f:
    Series18 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    Series18.device="cpu"

with open("../GP/Series19.pkl", "rb") as f:
    Series19 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    Series19.device="cpu"


# Define time series
time = np.linspace(0.0, 6000.0, 100)

# connect to the UM-Bridge model.
umbridge_model = umbridge.HTTPModel('http://localhost:4242', "forward")

# wrap the UM-Bridge model in the tinyDA UM-Bridge interface.
my_model0 = tda.UmBridgeModel(umbridge_model, umbridge_config={"level": 0})
my_model1 = tda.UmBridgeModel(umbridge_model, umbridge_config={"level": 1})
my_model2 = tda.UmBridgeModel(umbridge_model, umbridge_config={"level": 2})

# set up the prior.
my_prior = uniform_2D(-200000, -200000, 400000, 400000)

# Read the Probe Data
probe18 = pd.read_csv("probes/21418_march.csv", header=None)
probe19 = pd.read_csv("probes/21419_march.csv", header=None)

probe18 = processBuoyData(probe18)
probe19 = processBuoyData(probe19)

# Standardize scale
ssha18 = Series18._scale_outputs(np.interp(time, probe18["t"], probe18["ssha"]).reshape(-1, 1)).flatten()
ssha19 = Series19._scale_outputs(np.interp(time, probe19["t"], probe19["ssha"]).reshape(-1, 1)).flatten()

# set the likelihood
data_l0 = np.array([ssha18, ssha19])
data_l1 = np.array([ssha18, ssha19])
data_l2 = np.array([ssha18, ssha19])

# unscaled min18: -2.297739, ,unscaled min19: -0.551974
shift18 = Series18._scale_outputs(np.array([[-2.297739]])).flatten()
shift19 = Series19._scale_outputs(np.array([[-0.551974]])).flatten()

my_loglike_l0 = WassersteinLoglikeLinear(data_l0, np.concatenate([shift18, shift19]), 5000)
my_loglike_l1 = WassersteinLoglikeLinear(data_l1, np.concatenate([shift18, shift19]), 5000)
my_loglike_l2 = WassersteinLoglikeLinear(data_l2, np.concatenate([shift18, shift19]), 5000)

# initialise the LinkFactory
# the umbridge model contains three levels 

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

iterations = 30

# Initialise chain
my_chains = tda.sample(my_posteriors, my_proposal, iterations=iterations, n_chains=5, subchain_length=[100, 20], force_sequential=False)


import arviz as az
import pickle

with open("FWI_tinyda.pkl", "wb") as f:
    pickle.dump(my_chains, f)

# convert the tinyDA chains to an ArViz InferenceData object.
idata = tda.to_inference_data(my_chains, burnin=1000, level="0")

# display posterior summary statistics.
print(az.summary(idata))




