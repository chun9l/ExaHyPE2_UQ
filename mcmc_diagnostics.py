import tinyDA as tda 
import arviz as az
import pickle 
import matplotlib.pyplot as plt

pkl_path = "tinyda_result/tinyda.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)
    
levels = ["0", "1", "2"]

for i in levels:
    if i == "0":
        burnin = 1000
    else:
        burnin = 1 # Remove the first sample which serves as an inital guess in tinyDA
    print("Processing tinyDA chain data on level {}".format(i))
    idata = tda.to_inference_data(data, level=i, burnin=burnin)
    print(az.summary(idata))
    az.plot_trace(idata, compact=False, show=True)
    az.plot_pair(idata, var_names=["x0", "x1"], kind="kde", marginals=True, show=True)