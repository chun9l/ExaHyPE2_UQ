import matplotlib.pyplot as plt
from datetime import datetime
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pickle

DATETIME_FMT = "%H:%M:%S.%f %d/%m/%Y"

def set_fill_color(bp, color):
    plt.setp(bp["boxes"], color=color)

def parse_log_file(filepath):
    df = pd.read_csv(
        filepath,
        delimiter=",",
        skiprows=2,
        header=None,
        names=["process", "level", "chain_id", "start", "end"]
    )

    df["start"] = pd.to_datetime(df["start"], format=DATETIME_FMT)
    df["end"]   = pd.to_datetime(df["end"],   format=DATETIME_FMT)

    return df


def load_node_logs(log_files):
    """ Load multiple node log files.
    log_files is a dictionary with the node id as key 
    and the path to the file as the value.
    """    
    return {node: parse_log_file(path) for node, path in log_files.items()}


def combine_node_logs(log_files):
    """
    Load multiple log files and concat
    into pandas DataFrame
    """
    dfs = []

    for slurm_id, path in log_files.items():
        df = parse_log_file(path)
        df["slurm_id"] = slurm_id
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined

def compute_elapsed(df):
    """
    The time in the raw logs are given in datetime format 
    as in DATETIME_FMT. This converts them to elapsed time in minutes.
    """
    reference = df["start"].min() # Find the earliest start time

    df["start_sec"] = (df["start"] - reference).dt.total_seconds()
    df["end_sec"] = (df["end"] - reference).dt.total_seconds() 
    return df, reference

def plot(
    log_files=None,
    pkl_file=None,
    by_level=False,
    show_legend=True,
    show=False,
    save_path=None,
):
    plot_title = ["Request Idle Time"]
    plot_data = {}

    # Load logs
    if log_files:
        node_data = combine_node_logs(log_files)
        node_data, ref_dt = compute_elapsed(node_data)

    if pkl_file and by_level:
        plot_data_sacct = {}
        with open(pkl_file, "rb") as h:
            sacct_data = pickle.load(h)

        df_sacct = pd.DataFrame.from_dict(sacct_data, orient="index").reset_index().rename(columns={"index": "id"})
        df_sacct = df_sacct.iloc[15:].reset_index(drop=True)
        conditions = [
	    df_sacct['cpu-time'] < 10,
	    (df_sacct['cpu-time'] >= 10) & (df_sacct['cpu-time'] < 25),
	    (df_sacct['cpu-time'] >= 25) & (df_sacct['cpu-time'] < 60)
	]
        choices = [0, 1, 2]
        
        df_sacct['level'] = np.select(conditions, choices, default=np.nan)

    idle_time = []
    if log_files and by_level:
        idle_time0 = []
        idle_time1 = []
        idle_time2 = []
        makespan0 = []
        makespan1 = []
        makespan2 = []
        makespan = []
        plot_title.append("Makespan")

    # Get idle time based on what's on x axis
    if log_files:
        for (node, grp) in node_data.groupby("slurm_id"):
            start = grp["start_sec"]
            end = grp["end_sec"]
            grp["delay_sec"] = start - end.shift(1)
            if by_level:
                idle_time0 += (grp[grp["level"] == 0]["delay_sec"].to_list()[1:])
                idle_time1 += (grp[grp["level"] == 1]["delay_sec"].to_list()[1:])
                idle_time2 += (grp[grp["level"] == 2]["delay_sec"].to_list()[1:])
                grp["makespan"] = end - start
                makespan0 += (grp[grp["level"] == 0]["makespan"].to_list())
                makespan1 += (grp[grp["level"] == 1]["makespan"].to_list())
                makespan2 += (grp[grp["level"] == 2]["makespan"].to_list())
            else:
                idle_time.append(grp["delay_sec"].to_list()[1:])
    
    if log_files and by_level:
        idle_time.extend((idle_time0, idle_time1, idle_time2))
        makespan.extend((makespan0, makespan1, makespan2))
        plot_data["Makespan"] = makespan

    plot_data["Request Idle Time"] = idle_time
    
    if pkl_file and by_level:
        for m in plot_title:
            if m == "Makespan":
                metric = "makespan"
                plot_data_sacct[m] = []
            elif m == "Request Idle Time":
                metric = "lag"
                plot_data_sacct[m] = []
            
            plot_data_sacct[m].extend([df_sacct[df_sacct["level"] == i][metric] for i in range(0, 3)])
            

    for title in plot_title:
        fig, ax = plt.subplots(constrained_layout=True)
        fig.suptitle(title)

        # Plot boxplots
        if pkl_file and by_level:
            ppam = ax.boxplot(plot_data[title], positions=np.array(range(len(plot_data[title]))) * 2 - 0.4 * int(by_level), meanline=True,
                           showmeans=True, meanprops={"linestyle": "--", "color": "black", "linewidth": "1.5"},
                           medianprops={"linestyle": "-", "color": "black", "linewidth": "1.5"})
            set_fill_color(ppam, "blue")

            isc = ax.boxplot(plot_data_sacct[title], positions=np.array(range(len(plot_data_sacct[title]))) * 2.0 + 0.4 * int(by_level), meanline=True, showmeans=True,
                        meanprops={"linestyle": "--", "color": "black", "linewidth": "1.5"},
                        medianprops={"linestyle": "-", "color": "black", "linewidth": "1.5"})
            set_fill_color(isc, "red")
            ax.set_xticks(range(0, len(plot_data[title]) * 2, 2))
            ax.plot([], 'b-', linewidth=1, label="Persistent allocation")
            ax.plot([], 'r-', linewidth=1, label="Per-request allocation")
        elif pkl_file == None and by_level:
            ppam = ax.boxplot(plot_data[title], meanline=True,
                           showmeans=True, meanprops={"linestyle": "--", "color": "black", "linewidth": "1.5"},
                           medianprops={"linestyle": "-", "color": "black", "linewidth": "1.5"})
            
        else:
            ppam = ax.boxplot(plot_data[title], meanline=True,
                           showmeans=True, meanprops={"linestyle": "--", "color": "black", "linewidth": "1.5"},
                           medianprops={"linestyle": "-", "color": "black", "linewidth": "1.5"})

        ax.plot([], 'k-', linewidth=1.5, label="Median")
        ax.plot([], 'k--', linewidth=1.5, label="Mean")
        ax.plot([], 'ko', markerfacecolor='white', label="Fliers")
        ax.set_yscale("log")

        if by_level:
            x_labels = list(node_data["level"].unique())
            ax.set_xticklabels(sorted(x_labels))
            ax.set_xlabel("Level")
        else:
            x_labels = [f"Task {i}" for i in range(len(list(node_data["slurm_id"].unique())))]
            ax.set_xticklabels(x_labels)

        ax.set_ylabel(f"Time [s]")
        
        if show_legend:
            ax.legend(
                fontsize=7,
                framealpha=0.65,
            )
            
        if save_path:
            fig.savefig(save_path + os.sep + f"{title}.pdf", dpi=300, bbox_inches="tight")

        if show:
            plt.show()

    return fig
        
file_path = "../results/" + os.sep
pkl_file = file_path +  "slurm_um.pkl"
slurm_job_id = "10956759"
log_files = {f"{slurm_job_id}_{i}": f"{file_path + slurm_job_id + os.sep + slurm_job_id}_{i}/active_time.log" for i in range(1, 6)}
plot(
    log_files,
    pkl_file=None,
    by_level            = True, # Plot box per level to compare ISC
    # show                = True,
    show_legend         = True,
    save_path           = "/nobackup/mghw54/ExaHyPE2_UQ/plotting/",
)
