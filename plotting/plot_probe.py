import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import matplotlib.lines as mlines
import pickle
import xarray
import tinyDA as tda
import torch
import arviz as az
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/ming/ExaHyPE2_UQ/GP/')
from weighted_gp import WeightedGP

time = np.linspace(0.0, 6000.0, 100)

with open("../GP/Series18.pkl", "rb") as f:
    Series18 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    Series18.device = "cpu"

with open("../GP/Series19.pkl", "rb") as f:
    Series19 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    Series19.device = "cpu"

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def processBuoyData(buoy, cutoff=0.0035):
    processed = pd.DataFrame()
    processed["t"]= pd.to_numeric(59 + buoy[2]+ buoy[3]/float(24)+ buoy[4]/float(24*60) + buoy[5]/float(24*3600) -70.2422) *24 * 60 * 60
    processed["ssha"]= pd.to_numeric(buoy[8])

    processed = processed[processed["t"].between(0.0, 6000.0, inclusive="both")]
    
    if cutoff != None:
        sample_rate = len(processed["t"]) / (max(processed["t"]) - min(processed["t"]))
        processed["ssha"] = lowpass(processed["ssha"].values, cutoff, sample_rate)
        
    return processed

def plotProbe(buoy, samples=None, buoy_name=None):
    fig_width = 10.0
    fig_height = fig_width * (np.sqrt(2)-1.0)/2.0
    figsize = (fig_width, fig_height)
    fig, ax = plt.subplots(figsize=figsize)
    
    x_buoy = buoy['t'] / 60.0
    y_buoy = buoy['ssha']
    
    ax.plot(x_buoy,y_buoy,linewidth=2, markersize=5,marker="x",color='r')
    handles = [mlines.Line2D([], [], color='r', linestyle='-',label="Buoy {}".format(buoy_name))]

    if samples != None:
        combined = az.extract_dataset(samples.posterior, num_samples=50)
        posterior_df = xarray.Dataset.to_pandas(combined)
        x_gp = time / 60.0
        if buoy_name == "21418":
            GP = Series18
        elif buoy_name == "21419":
            GP = Series19
        
        for index, row in posterior_df.iterrows():
            x = row["x0"]
            y = row["x1"]
            series = np.hstack([np.repeat([[x, y]], 100, axis=0), time.reshape(-1, 1)]) 
            y_gp = GP.predict(series)[0]
            
            ax.plot(x_gp,y_gp.detach().numpy(),linewidth=1,alpha=0.1,color='k')
        handles += [mlines.Line2D([], [], color='k', linestyle='-',label='l0 samples')]
    
    ax.legend(handles=handles)
    fig.tight_layout()
    
    fig.subplots_adjust(bottom=0.25)
    fig.subplots_adjust(left=0.05)

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("ssha [m]")  
    
    plt.show()
    return fig,ax

buoy18 = pd.read_csv("../probes/21418_march.csv", header=None)
buoy19 = pd.read_csv("../probes/21419_march.csv", header=None)

buoy18 = processBuoyData(buoy18, None)
buoy19 = processBuoyData(buoy19, None)

with open("../tinyda_results/tinyda.pkl", "rb") as f:
    data = pickle.load(f)

idata = tda.to_inference_data(data, burnin=1000, level="0")

plotProbe(buoy18, samples=idata, buoy_name="21418")
plotProbe(buoy19, samples=idata, buoy_name="21419")
