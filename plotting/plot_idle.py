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
    reference = df["start"].min()

    df["start_min"] = (df["start"] - reference).dt.total_seconds() / 60
    df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60

    return df, reference

def plot(
    log_files,
    *,
    title="Boxplots of idle time",
    show_legend=True,
    show=False,
    save_path=None,
):
    
    # Load logs
    node_data = combine_node_logs(log_files)
    node_data, ref_dt = compute_elapsed(node_data)

    fig, ax = plt.subplots(constrained_layout=True)
    
    # Get idle time between requests
    for y_idx, (node, grp) in enumerate(df.groupby("slurm_id")):
        starts = grp["start_min"].to_numpy()
        widths = grp["duration_min"].to_numpy()

    # Plot boxplots
    hq = ax.boxplot(data_hq, positions=np.array(range(len(data_hq))) * 2.0 + 0.4, meanline=True, showmeans=True,
                        meanprops={"linestyle": "--", "color": "black", "linewidth": "1.5"},
                        medianprops={"linestyle": "-", "color": "black", "linewidth": "1.5"})
    ax.set_xticks(range(0, len(benchmark) * 2, 2), benchmark)
    plt.plot([], 'b-', linewidth=1, label="SLURM")
    plt.plot([], 'r-', linewidth=1, label="HQ")
    plt.plot([], 'k-', linewidth=1.5, label="Median")
    plt.plot([], 'k--', linewidth=1.5, label="Mean")
    plt.plot([], 'ko', markerfacecolor='white', label="Fliers")
    ax.set_yscale("log")

    node_labels = list(node_data["slurm_id"].unique())
    
    ax.set_yticks(range(len(node_labels)))
    ax.set_yticklabels([f"Task {i}" for i in range(len(node_labels))])

    ax.set_xlabel(
        f"Elapsed time (minutes)\n[reference: {ref_dt.strftime(DATETIME_FMT)}]"
    )
    
    plt.plot([], 'k-', linewidth=1.5, label="Median")
    plt.plot([], 'k--', linewidth=1.5, label="Mean")
    plt.plot([], 'ko', markerfacecolor='white', label="Fliers")

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

os.chdir("/home/ming/UMBridge_Loadbalancer/")
for i in range(len(metrics)):
    for job in job_count:
        fig, ax = plt.subplots()
        fig.suptitle(f"{metrics[i]} {job} jobs")
        if i != 3:
            ax.set_ylabel("Time (s)")
        else:
            ax.set_ylabel("Arbitrary units")
        data_slurm = [data_dict[scheduler[0].lower()][job][metrics[i]][app] for app in benchmark]
        data_hq = [data_dict[scheduler[1].lower()][job][metrics[i]][app] for app in benchmark]
        slurm = ax.boxplot(data_slurm, positions=np.array(range(len(data_slurm))) * 2 - 0.4, meanline=True,
                           showmeans=True, meanprops={"linestyle": "--", "color": "black", "linewidth": "1.5"},
                           medianprops={"linestyle": "-", "color": "black", "linewidth": "1.5"})
        hq = ax.boxplot(data_hq, positions=np.array(range(len(data_hq))) * 2.0 + 0.4, meanline=True, showmeans=True,
                        meanprops={"linestyle": "--", "color": "black", "linewidth": "1.5"},
                        medianprops={"linestyle": "-", "color": "black", "linewidth": "1.5"})
        ax.set_xticks(range(0, len(benchmark) * 2, 2), benchmark)
        set_fill_color(slurm, "blue")
        set_fill_color(hq, "red")
        plt.plot([], 'b-', linewidth=1, label="SLURM")
        plt.plot([], 'r-', linewidth=1, label="HQ")
        plt.plot([], 'k-', linewidth=1.5, label="Median")
        plt.plot([], 'k--', linewidth=1.5, label="Mean")
        plt.plot([], 'ko', markerfacecolor='white', label="Fliers")
        if metrics[i] != "SLR":  ax.set_yscale("log")
        plt.legend()
        # plt.show()
        # plt.savefig(f"{metrics[i]}_{job}.pdf", format="pdf")
        
if __name__ == "__main__":
    slurm_job_id = "10956759"
    file_path = "../results/" +  slurm_job_id + os.sep
    log_files = {f"{slurm_job_id}_{i}": f"{file_path + slurm_job_id}_{i}/active_time.log" for i in range(1, 6)}
    plot(
        log_files,
        show_dep            = True,
        arrow_chains        = [2],
        title               = "Process Uptime",
        # show                = True,
        show_legend         = False,
        save_path           = f"uptime_{slurm_job_id}.png",
    )