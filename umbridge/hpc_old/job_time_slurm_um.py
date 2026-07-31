import json
import os 
import subprocess
from dateutil import parser
import pickle
import glob
import re

def extract_times(save_dir, starttime, endtime):  
    if os.path.isfile(f"{save_dir}slurm_um.pkl"):
        print(f"{save_dir}slurm_um.pkl exists! Skipping pkl creation")
        return 
    else:
        main_dir = f"{save_dir}"
        cmd = ["sacct", f"--starttime={str(starttime)}", f"--endtime={str(endtime)}", "--json"]
        output = subprocess.run(cmd, stdout=subprocess.PIPE)
        json_data = json.loads(output.stdout.decode("utf-8"))
    
    data = {}
    for job in json_data["jobs"]:
        job_id = job["job_id"]
        cpu_cores = int(job["required"]["CPUs"])
        submit = job["time"]["submission"]
        start = job["time"]["start"]
        end = job["time"]["end"]
        if len(job["steps"]) == 2:
            batch = job["steps"][0]["time"]["total"]["seconds"] + job["steps"][0]["time"]["total"]["microseconds"] / 1e6
            extern = job["steps"][1]["time"]["total"]["seconds"] + job["steps"][1]["time"]["total"]["microseconds"] / 1e6
            job_steps = (batch + extern)
        else:
            raise Exception("Incorrect job steps")
        makespan = end - submit
        if makespan == 0:
            makespan += job_steps
            lag = 0
        else: 
            lag = end - submit - job_steps
        slr = makespan / job_steps
        try:
            data[str(job_id)] = {"makespan": makespan, "cpu-time": job_steps, "lag": lag, "slr": slr}
        except:
            print(job_id, submit, start, end, job_steps)

    with open(f"{save_dir}slurm_um.pkl", "wb") as h:
        pickle.dump(data, h)


save_dir = "/nobackup/mghw54/ExaHyPE2_UQ/results/"
extract_times(save_dir, "2026-07-30T23:00:00", "2026-07-31T08:00:00")
