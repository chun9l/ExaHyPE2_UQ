import matplotlib.pyplot as plt
from datetime import datetime
import os
from datetime import timedelta
import pandas as pd

DATETIME_FMT = "%H:%M:%S.%f %d/%m/%Y"

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
    """
    Load multiple node log files.
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
    log_files,
    *,
    title=None,
    show_legend=True,
    show=False,
    save_path=None,
):
    
    # Load logs
    node_data = combine_node_logs(log_files)
    node_data, ref_dt = compute_elapsed(node_data)

    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle(title)
    
    idle_time = []
    # Get idle time between requests
    for y_idx, (node, grp) in enumerate(node_data.groupby("slurm_id")):
        start = grp["start_sec"]
        end = grp["end_sec"]
        
        delay = start - end.shift(1)
        idle_time.append(delay.to_numpy()[1:])

    # Plot boxplots
    hq = ax.boxplot(idle_time, meanline=True, showmeans=True,
                        meanprops={"linestyle": "--", "color": "black", "linewidth": "1.5"},
                        medianprops={"linestyle": "-", "color": "black", "linewidth": "1.5"})

    ax.plot([], 'k-', linewidth=1.5, label="Median")
    ax.plot([], 'k--', linewidth=1.5, label="Mean")
    ax.plot([], 'ko', markerfacecolor='white', label="Fliers")
    ax.set_yscale("log")

    node_labels = list(node_data["slurm_id"].unique())
    
    ax.set_xticks(range(1, len(node_labels) + 1))
    ax.set_xticklabels([f"Task {i}" for i in range(len(node_labels))])

    ax.set_ylabel(
        f"Idle time [s]"
    )
    
    if show_legend:
        ax.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.65,
        )
        
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    if show:
        plt.show()

    return fig
        
if __name__ == "__main__":
    slurm_job_id = "10956759"
    file_path = "../results/" +  slurm_job_id + os.sep
    log_files = {f"{slurm_job_id}_{i}": f"{file_path + slurm_job_id}_{i}/active_time.log" for i in range(1, 6)}
    plot(
        log_files,
        title               = "Process idle time",
        # show                = True,
        show_legend         = True,
        save_path           = f"idle_time_{slurm_job_id}.svg",
    )