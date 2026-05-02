import umbridge
import os
import json
import numpy as np
import pandas as pd
import torch
from GP.weighted_gp import WeightedGP
import datetime
import csv

def processExahype2Data(num):
    num["ssha"] = pd.to_numeric(num["data(0)"] + num["data(3)"])
    return num

def getQOI(num):
    ssha = num["ssha"].values
    time = num["t"].values
    return time, ssha

# Load GP Model. Change to your path
gp_dir = "/nobackup/mghw54/ExaHyPE2_UQ/GP/"

with open(f"{gp_dir}Series18.pkl", "rb") as f:
    Series18 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    Series18.device = "cpu"

with open(f"{gp_dir}Series19.pkl", "rb") as f:
    Series19 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    Series19.device = "cpu"

class ExahypeModel(umbridge.Model):
    def __init__(self, logging=False):
        super().__init__("forward")
        self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")
        self.request = 0
        self.logging = logging
        self.time = np.linspace(0.0, 6000.0, 100) # Define time series
        try:
            self.slurm_id = str(os.environ["SLURM_ARRAY_JOB_ID"])
            self.job_arr_id = str(os.environ["SLURM_ARRAY_TASK_ID"])
        except:
            print("Not in SLURM evironment")
            self.slurm_id = "0"
            self.job_arr_id = "0"

        self.source_dir = f"/nobackup/mghw54/Peano/applications/shallow-water/tohoku-tsunami/"
        self.output_dir = "/nobackup/mghw54/Peano/applications/shallow-water/tohoku-tsunami" + os.sep + str(self.slurm_id) + "_" + str(self.job_arr_id) + "/"
        self.input_file = "tohoku-tsunami.json" # Select input file 
        
        os.system(f"mkdir -p {self.output_dir}")

        if self.logging == True:
            print("Logging enabled")
            self.active_time_log = open(self.output_dir + os.sep + "active_time.log", "a")
            header = ["request", "level", "chain_id", "start_time", "end_time"]
            self.writer = csv.writer(self.active_time_log, delimiter=",")
            self.writer.writerow(header)
            self.writer.writerow([self.request, "None", "None", self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
            self.active_time_log.flush()

    def get_input_sizes(self, config):
        return [2] 

    def get_output_sizes(self, config):
        return [100, 100] 

    def process_parameters(self, parameters):
        parameters = np.repeat(parameters, 100, axis=0)
        return np.hstack([parameters, self.time.reshape(-1, 1)])

    def __call__(self, parameters, config):
        level = str(config.get("level"))
        chain_id = str(config.get("chain_id"))
        
        if level not in ["0", "1", "2"]:
            raise Exception("level must be 0, 1 or 2")

        parameters = self.process_parameters(parameters)

        # Run the model 
        if level == "0":
            if self.logging == True:
                self.request += 1
                self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")

            ssha18, _ = Series18.predict(np.array(parameters), return_scaled=True)
            ssha19, _ = Series19.predict(np.array(parameters), return_scaled=True)

            if self.logging == True:
                self.writer.writerow([self.request, level, chain_id, self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
                self.active_time_log.flush()

            return [ssha18.detach().numpy().flatten().tolist(), ssha19.detach().numpy().flatten().tolist()]

        else:
            if self.logging == True:
                self.request += 1
                self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")

            source_dir = self.source_dir + f"l{level}/"
            os.system(f"cp {source_dir + self.input_file} {self.output_dir}")

            with open(self.output_dir + self.input_file, 'r+') as f: 
                data = json.load(f)
                data["displacementPosition"] = [float(parameters[0][0]), float(parameters[0][1])]
                data["plot"]["outputPath"] = self.output_dir
                f.seek(0)
                json.dump(data, f)
                f.truncate()

            os.chdir(source_dir)
            os.system(f"srun -n 2 -c 16 --mpi=pmix_v4  ./ExaHyPE --config-file {self.output_dir + self.input_file}")
        
        probe18_path = self.output_dir + os.sep + "Probes-rank-0.csv"
        probe19_path = self.output_dir + os.sep + "Probes-rank-1.csv"

        if os.path.isfile(probe18_path) and os.path.isfile(probe19_path):
            df18 = pd.read_csv(probe18_path, skipinitialspace=True)
            df19 = pd.read_csv(probe19_path, skipinitialspace=True)

            # sort according to time for interpolation
            df18.sort_values(by=["t"], inplace=True)
            df19.sort_values(by=["t"], inplace=True)

            df18 = processExahype2Data(df18)
            df19 = processExahype2Data(df19)

            time18, ssha18 = getQOI(df18)
            time19, ssha19 = getQOI(df19)

            # interpolate data 
            y18 = Series18._scale_outputs(np.interp(parameters[:, 2], time18, ssha18).reshape(-1, 1)).flatten()
            y19 = Series19._scale_outputs(np.interp(parameters[:, 2], time19, ssha19).reshape(-1, 1)).flatten()

            if self.logging == True:
                self.writer.writerow([self.request, level, chain_id, self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
                self.active_time_log.flush()

            return [y18.tolist(), y19.tolist()] 

        else:
            print("Incomplete simulation output!")
            return [[0.0, 0.0]]
             

    def supports_evaluate(self):
        return True

if "PORT" in os.environ:
    port = os.environ['PORT']
else:
    port = 4249

model = ExahypeModel(logging=True)
umbridge.serve_models([model], int(port))
