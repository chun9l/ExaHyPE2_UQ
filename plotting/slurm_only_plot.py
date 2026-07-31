import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

level = ["0", "1", "2"]
benchmark = "Predator-Prey"
metrics = ["Makespan", "CPUTime", "Scheduler Overhead", "SLR"]


def set_fill_color(bp, color):
    plt.setp(bp['boxes'], color=color)

def plot_boxplots(data_loc, save_loc):
    data_dict = {"0": {}, "1": {}, "2": {}}
    l0 = []
    l1 = []
    l2 = []
    with open(data_loc, "rb") as h:
        data = pickle.load(h)
    for m in metrics:
        if m == "Makespan":
            metric = "makespan"
        elif m == "CPUTime":
            metric = "cpu-time"
        elif m == "Scheduler Overhead":
            metric = "lag"
        elif m == "SLR":
            metric = "slr"
        
        data_dict["0"][m] = []
        data_dict["1"][m] = []
        data_dict["2"][m] = []

        for (i, job) in enumerate(data):
            if i < 15:
                continue
            else:
                if data[job]["cpu-time"] and data[job]["cpu-time"] < 10.0:
                    data_dict["0"][m].append(data[job][metric])
                elif 10.0 <= data[job]["cpu-time"] and data[job]["cpu-time"] < 25.0:
                    data_dict["1"][m].append(data[job][metric])
                elif 25.0 <= data[job]["cpu-time"] and data[job]["cpu-time"] < 60.0:
                    data_dict["2"][m].append(data[job][metric])

    for i in range(len(metrics)):
        fig, ax = plt.subplots()
        fig.suptitle(f"{metrics[i]}")
        if i != 3:
            ax.set_ylabel("Time (s)")
        else:
            ax.set_ylabel("Arbitrary units")
        data_slurm_um = [data_dict[l][metrics[i]] for l in level]
        slurm_um = ax.boxplot(data_slurm_um, meanline=True, showmeans=True,
                        meanprops={"linestyle": "--", "color": "black", "linewidth": "1.5"},
                        medianprops={"linestyle": "-", "color": "black", "linewidth": "1.5"})
        ax.set_xticks(range(1, len(level) + 1))
        ax.set_xticklabels([f"Level {i}" for i in level])
        plt.plot([], 'k-', linewidth=1.5, label="Median")
        plt.plot([], 'k--', linewidth=1.5, label="Mean")
        plt.plot([], 'ko', markerfacecolor='white', label="Fliers")
        if metrics[i] != "SLR":  ax.set_yscale("log")
        plt.legend()
        # plt.show()
        plt.savefig(f"{save_loc}/{metrics[i]}_slurm_um.pdf", format="pdf")
        plt.plot([], 'b-', linewidth=1, label="SLURM")

data_loc = "/nobackup/mghw54/ExaHyPE2_UQ/results/slurm_um.pkl"
save_loc = "/nobackup/mghw54/ExaHyPE2_UQ/plotting/"

plot_boxplots(data_loc, save_loc)
