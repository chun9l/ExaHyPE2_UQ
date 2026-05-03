import umbridge
import csv
import os
import json
import datetime
import numpy as np
import pandas as pd
import torch
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/ming/ExaHyPE2_UQ/GP/')
from weighted_gp import WeightedGP

def processExahype2Data(num):
    num["ssha"] = pd.to_numeric(num["data(0)"] + num["data(3)"])
    return num

def getQOI(num):
    max_id = num["ssha"].idxmax()
    max_row = num.loc[max_id]
    return (max_row["t"], max_row["ssha"])

# Load GP Model. Change to your path
gp_dir = "/cosma/apps/do009/dc-loi1/ExaHyPE2_UQ/GP/"

with open(f"{gp_dir}Time18.pkl", "rb") as f:
    Time18 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    Time18.device = "cpu"

with open(f"{gp_dir}Time19.pkl", "rb") as f:
    Time19 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    Time19.device = "cpu"

with open(f"{gp_dir}SSHA18.pkl", "rb") as f:
    SSHA18 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    SSHA18.device = "cpu"

with open(f"{gp_dir}SSHA19.pkl", "rb") as f:
    SSHA19 = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    SSHA19.device = "cpu"

class ExahypeModel(umbridge.Model):
    def __init__(self, logging=False):
        super().__init__("forward")
        self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")
        self.request=0
        self.logging = logging
        self.slurm_id = str(os.getenv("SLURM_ARRAY_JOB_ID", 0))
        self.job_arr_id = str(os.getenv("SLURM_ARRAY_TASK_ID", 0))
        self.source_dir = f"/cosma/apps/do009/dc-loi1/Peano/applications/shallow-water/tohoku-tsunami/"
        self.output_dir = f"/cosma/apps/do009/dc-loi1/Peano/applications/shallow-water/tohoku-tsunami/" + os.sep + str(self.slurm_id) + "_" + str(self.job_arr_id) + "/"
        self.input_file = "tohoku-tsunami.json" # Select input file 
        
        os.system(f"mkdir -p {self.output_dir}")

        if self.logging == True:
            print("Logging enabled")
            self.active_time_log = open(self.output_dir + os.sep + "active_time.log", "a")
            header = ["request", "level", "chain_id", "start_time", "end_time"]
            self.writer = csv.writer(self.active_time_log, delimiter=',')
            self.writer.writerow(header)
            self.writer.writerow([self.request, "None", "None", self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
            self.active_time_log.flush()

    def get_input_sizes(self, config):
        return [2] 

    def get_output_sizes(self, config):
        return [4] 

    def __call__(self, parameters, config):
        level = str(config.get("level"))
        chain_id = str(config.get("chain_id"))
        
        if level not in ["0", "1", "2"]:
            raise Exception("level must be 0, 1 or 2")

        # Run the model 
        if level == "0":
            if self.logging == True:
                self.request += 1
                self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")

            time18, _ = Time18.predict(np.array([parameters[0]]))
            time19, _ = Time19.predict(np.array([parameters[0]]))
            ssha18, _ = SSHA18.predict(np.array([parameters[0]]))
            ssha19, _ = SSHA19.predict(np.array([parameters[0]]))
            if self.logging == True:
                self.writer.writerow([self.request, level, chain_id, self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
                self.active_time_log.flush()

            return [[time18[0][0].item() / 60.0, time19[0][0].item() / 60.0, ssha18[0][0].item(), ssha19[0][0].item()]]

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
            os.system(f"OMP_NUM_THREADS=16 mpirun -n 2 --map-by numa:PE=16 ./ExaHyPE --config-file {self.output_dir + self.input_file}")
        
        probe18_path = self.output_dir + os.sep + "Probes-rank-0.csv"
        probe19_path = self.output_dir + os.sep + "Probes-rank-1.csv"

        if os.path.isfile(probe18_path) and os.path.isfile(probe19_path):
            df18 = pd.read_csv(probe18_path, skipinitialspace=True)
            df19 = pd.read_csv(probe19_path, skipinitialspace=True)

            df18 = processExahype2Data(df18)
            df19 = processExahype2Data(df19)

            time18, ssha18 = getQOI(df18)
            time19, ssha19 = getQOI(df19)
            
            if self.logging == True:
                self.writer.writerow([self.request, level, chain_id, self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
                self.active_time_log.flush()

            return [[time18 / 60.0, time19 / 60.0, ssha18, ssha19]] 

        else:
            print("Incomplete simluation output!")
            return [[0.0, 0.0, 0.0, 0.0]]

    def supports_evaluate(self):
        return True

if "PORT" in os.environ:
    port = os.environ['PORT']
else:
    port = 4249

model = ExahypeModel(logging=True)
umbridge.serve_models([model], int(port))

