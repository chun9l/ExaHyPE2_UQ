import datetime
import csv
import numpy as np
from scipy.integrate import solve_ivp
import umbridge
import os
import time
import random

np.random.seed(0)
random.seed(0)

log_flag = True

request = 0
t_eval = np.linspace(0, 12, 25)

class PredatorPreyModel_l0(umbridge.Model):
    def __init__(self, logging=False):
        super().__init__("l0")
        self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")
        self.logging = logging
        self.slurm_id = str(os.getenv("SLURM_ARRAY_JOB_ID", 0))
        self.job_arr_id = str(os.getenv("SLURM_ARRAY_TASK_ID", 0)) 
        self.output_dir = "/nobackup/mghw54/ExaHyPE2_UQ/results" + os.sep + str(self.slurm_id) + os.sep + str(self.slurm_id) + "_" + str(self.job_arr_id) + "/"

        os.system(f"mkdir -p {self.output_dir}")

        if self.logging == True:
            print("Logging enabled")
            self.active_time_log = open(self.output_dir + os.sep + "active_time.log", "a")
            header = ["request", "level", "chain_id", "start_time", "end_time"]
            self.writer = csv.writer(self.active_time_log, delimiter=',')
            self.writer.writerow(header)
            self.writer.writerow([request, "None", "None", self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
            self.active_time_log.flush()
        
        
        # set the timesteps, where we are collecting the model output
        self.datapoints = t_eval[: 8]
        
        # set the span of the integration.
        self.t_span = [0, self.datapoints[-1]]
        
    def get_input_sizes(self, config):
        return [6]

    def get_output_sizes(self, config):
        return [16]

    def supports_evaluate(self):
        return True

    def dydx(self, t, y, a, b, c, d):
        # Lotka-Volterra Model model, see e.g. https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
        return np.array([a*y[0] - b*y[0]*y[1], c*y[0]*y[1] - d*y[1]])

    def __call__(self, parameters, config={}):
        global request
        
        level = 0
        chain_id = str(config.get("chain_id"))
        
        # extract the parameters, and take the exponential to keep them positive
        P_0, Q_0, a, b, c, d = np.exp(parameters[0])
        
        # solve the initial value problem.
        if self.logging == True:
            request += 1
            self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")

        self.y = solve_ivp(lambda t, y: self.dydx(t, y, a, b, c, d), self.t_span, np.array([P_0, Q_0]), t_eval=self.datapoints) 

        # Artificial workload
        A = np.random.rand(300, 300)
        end_time = time.time() + random.randint(0, 5)
        while time.time() < end_time:
            np.linalg.eig(A)

        if self.logging == True and chain_id != "None":
            self.writer.writerow([request, level, chain_id, self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
            self.active_time_log.flush()

        
        # return the results, only if the integration succeeded.
        if self.y.success:
            return [self.y.y.flatten().tolist()]
        else:
            return np.nan, False

class PredatorPreyModel_l1(umbridge.Model):
    def __init__(self, logging=False):
        super().__init__("l1")
        self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")
        self.logging = logging
        self.slurm_id = str(os.getenv("SLURM_ARRAY_JOB_ID", 0))
        self.job_arr_id = str(os.getenv("SLURM_ARRAY_TASK_ID", 0)) 
        self.output_dir = "/nobackup/mghw54/ExaHyPE2_UQ/results" + os.sep + self.slurm_id + os.sep + str(self.slurm_id) + "_" + str(self.job_arr_id) + "/"

        if self.logging == True:
            print("Logging enabled")
            self.active_time_log = open(self.output_dir + os.sep + "active_time.log", "a")
            self.writer = csv.writer(self.active_time_log, delimiter=',')

        # set the timesteps, where we are collecting the model output
        self.datapoints = t_eval[: 16]
        
        # set the span of the integration.
        self.t_span = [0, self.datapoints[-1]]
        
    def get_input_sizes(self, config):
        return [6]

    def get_output_sizes(self, config):
        return [32]

    def supports_evaluate(self):
        return True

    def dydx(self, t, y, a, b, c, d):
        # Lotka-Volterra Model model, see e.g. https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
        return np.array([a*y[0] - b*y[0]*y[1], c*y[0]*y[1] - d*y[1]])

    def __call__(self, parameters, config={}):
        global request

        level = 1
        chain_id = str(config.get("chain_id"))

        # extract the parameters, and take the exponential to keep them positive
        P_0, Q_0, a, b, c, d = np.exp(parameters[0])
        
        # solve the initial value problem.
        if self.logging == True:
            request += 1
            self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")

        self.y = solve_ivp(lambda t, y: self.dydx(t, y, a, b, c, d), self.t_span, np.array([P_0, Q_0]), t_eval=self.datapoints) 

        # Artificial workload
        A = np.random.rand(600, 600)
        end_time = time.time() + random.randint(10, 15)
        while time.time() < end_time:
            np.linalg.eig(A)

        if self.logging == True and chain_id != "None":
            self.writer.writerow([request, level, chain_id, self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
            self.active_time_log.flush()

        
        # return the results, only if the integration succeeded.
        if self.y.success:
            return [self.y.y.flatten().tolist()]
        else:
            return np.nan, False


class PredatorPreyModel_l2(umbridge.Model):
    def __init__(self, logging=False):
        super().__init__("l2")
        self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")
        self.logging = logging
        self.slurm_id = str(os.getenv("SLURM_ARRAY_JOB_ID", 0))
        self.job_arr_id = str(os.getenv("SLURM_ARRAY_TASK_ID", 0)) 
        self.output_dir = "/nobackup/mghw54/ExaHyPE2_UQ/results" + os.sep + self.slurm_id + os.sep + str(self.slurm_id) + "_" + str(self.job_arr_id) + "/"

        if self.logging == True:
            print("Logging enabled")
            self.active_time_log = open(self.output_dir + os.sep + "active_time.log", "a")
            self.writer = csv.writer(self.active_time_log, delimiter=',')

        # set the timesteps, where we are collecting the model output
        self.datapoints = t_eval
        
        # set the span of the integration.
        self.t_span = [0, self.datapoints[-1]]

    def get_input_sizes(self, config):
        return [6]

    def get_output_sizes(self, config):
        return [50]

    def supports_evaluate(self):
        return True

    def dydx(self, t, y, a, b, c, d):
        # Lotka-Volterra Model model, see e.g. https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
        return np.array([a*y[0] - b*y[0]*y[1], c*y[0]*y[1] - d*y[1]])

    def __call__(self, parameters, config={}):
        global request

        level = 2
        chain_id = str(config.get("chain_id"))

        # extract the parameters, and take the exponential to keep them positive
        P_0, Q_0, a, b, c, d = np.exp(parameters[0])
        
        # solve the initial value problem.
        if self.logging == True:
            request += 1
            self.start_time = datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")

        self.y = solve_ivp(lambda t, y: self.dydx(t, y, a, b, c, d), self.t_span, np.array([P_0, Q_0]), t_eval=self.datapoints) 

        # Artificial workload
        A = np.random.rand(1000, 1000)
        end_time = time.time() + random.randint(25, 50)
        while time.time() < end_time:
            np.linalg.eig(A)

        if self.logging == True and chain_id != "None":
            self.writer.writerow([request, level, chain_id, self.start_time, datetime.datetime.now().strftime("%H:%M:%S.%f %d/%m/%Y")])
            self.active_time_log.flush()

        
        # return the results, only if the integration succeeded.
        if self.y.success:
            return [self.y.y.flatten().tolist()]
        else:
            return np.nan, False

if "PORT" in os.environ:
    port = os.environ['PORT']
else:
    port = 4249

model_l0 = PredatorPreyModel_l0(logging=log_flag)
model_l1 = PredatorPreyModel_l1(logging=log_flag)
model_l2 = PredatorPreyModel_l2(logging=log_flag)
umbridge.serve_models([model_l0, model_l1, model_l2], int(port))
