import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


 
def compute_dependencies(df):
    """
    For each chain, sort row by the start time. It should then
    give the correct execution order within a chain since it is
    sequential.
    Each vertex = {
        'chain_id', 'from_level', 'to_level',
        'from_node', 'to_node',
        'from_end', 'to_start'
    }
    To show the request dependency, we just need to follow the sorted rows
    and keep track of the level as we go.
    """
    edges = []
    node_y_mapping = {node: y_idx for y_idx, (node, grp) in enumerate(df.groupby("slurm_id"))}

    for chain_id, grp in df.groupby("chain_id"):
        grp = grp.sort_values("start_min")
        rows = list(grp.itertuples(index=False))

        for i in range(len(rows) - 1):
            pred = rows[i]
            succ = rows[i + 1]

            edges.append({
                "chain_id": chain_id,
                "from_level": int(pred.level),
                "to_level": int(succ.level),
                "from_node": node_y_mapping[str(pred.slurm_id)],
                "to_node": node_y_mapping[str(succ.slurm_id)],
                "from_end": float(pred.start_min + pred.duration_min),
                "to_start": float(succ.start_min),
            })
    return edges


def compute_elapsed(df):
    """
    The time in the raw logs are given in datetime format 
    as in DATETIME_FMT. This converts them to elapsed time in minutes.
    """
    reference = df["start"].min()

    df["start_min"] = (df["start"] - reference).dt.total_seconds() / 60
    df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60

    return df, reference

 
def plot_barh(
    ax, 
    df, 
    color_map, 
    bar_height=0.6
):
    """
    Plots the horizontal bars using broken_barh
    """
    for y_idx, (node, grp) in enumerate(df.groupby("slurm_id")):
        starts = grp["start_min"].to_numpy()
        widths = grp["duration_min"].to_numpy()

        xranges = list(zip(starts, widths))

        colors = grp["level"].map(color_map).to_list()

        # One call draws ALL bars for this node
        ax.broken_barh(
            xranges,
            (y_idx - bar_height / 2, bar_height),
            facecolors=colors,
            edgecolors="white",
            linewidth=0,
        )

def plot_dependency_arrows(
    ax, 
    edges, 
    color_map, 
    arrow_chain_set=None, 
    arrow_level_set=None, 
    bar_height=0.6,
    min_gap=0.0,
    skip_arrows=1,
):
    """
    Plots the arrows that indicate evaluation dependency
    on the bar chart.
    """
 
    for edge in edges[::skip_arrows]: 
        if edge["chain_id"] not in arrow_chain_set:
            continue

        if arrow_level_set is not None and float(edge["from_level"]) not in arrow_level_set:
            continue
 
        gap = edge["to_start"] - edge["from_end"]
        if gap < min_gap:
            continue

        x_start = edge["from_end"]
        x_end = edge["to_start"]
        y_start = edge["from_node"]
        y_end = edge["to_node"]
        color = color_map[edge["chain_id"]]
 
        # Offset arrow endpoints slightly so they don't overlap bar edges
        offset = bar_height * 0.55

        dy = y_end - y_start
        rad = 0.2 if dy == 0 else min(0.5, 0.15 * abs(dy))
 
        if y_start == y_end:
            # Same row: horizontal arrow just above the bar
            y_arrow = y_start + offset * 0.8
            ax.annotate(
                "",
                xy=(x_end, y_arrow),
                xytext=(x_start, y_arrow),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=0.2,
                    connectionstyle=f"arc3,rad={rad}",
                ),
            )
        else:
            # Cross-row: diagonal arrow
            ax.annotate(
                "",
                xy=(x_end, y_end + (offset if y_end < y_start else -offset)),
                xytext=(x_start, y_start + (-offset if y_end < y_start else offset)),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=0.2,
                    connectionstyle=f"arc3,rad={rad}",
                ),
            )




def plot(
    log_files,
    *,
    show_dep=False,
    arrow_chains=None,
    arrow_levels=None,
    title="Process Uptime by Node",
    bar_height=0.6,
    figsize=None,
    show_legend=True,
    show=False,
    save_path=None,
):
    
    # Load logs
    node_data = combine_node_logs(log_files)
    node_data, ref_dt = compute_elapsed(node_data)
    edges = compute_dependencies(node_data)

    # Get MCMC levels in the logs
    all_levels = sorted(node_data["level"].unique())

    level_cmap = plt.get_cmap("tab10")
    level_color_map = {
        level: level_cmap(i % level_cmap.N)
        for i, level in enumerate(all_levels)
    }

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Plot bar 
    plot_barh(ax, node_data, level_color_map, bar_height=bar_height)

    # Plot dependency arrows
    if show_dep == True:
    # Only show arrows for specified levels
        if arrow_levels is not None:
            arrow_level_set = set(float(l) for l in arrow_levels)
        else:
            arrow_level_set = None  # None means "all levels"
        
    # One show arrows for specified chains
        all_chains = sorted(node_data["chain_id"].unique())

        if arrow_chains is not None:
            arrow_chain_set = set(c for c in arrow_chains)
        else:
            arrow_chain_set = set(all_chains)  # default: all chains

        chain_cmap = plt.get_cmap("Set2")
        chain_color_map = {
            chain_id: chain_cmap(i % chain_cmap.N)
            for i, chain_id in enumerate(all_chains)
        }
        plot_dependency_arrows(ax, edges, chain_color_map, arrow_chain_set, arrow_level_set, bar_height=bar_height, min_gap=0.0, skip_arrows=1000)


    node_labels = list(node_data["slurm_id"].unique())
    ax.set_yticks(range(len(node_labels)))
    ax.set_yticklabels(node_labels)
    ax.set_ylim(-0.5, len(node_labels) - 0.5)
    ax.invert_yaxis()

    ax.set_xlabel(
        f"Elapsed time (minutes)\n[reference: {ref_dt.strftime(DATETIME_FMT)}]"
    )
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Configure legend
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    level_patches = [
        mpatches.Patch(color=level_color_map[level], label=f"Level {level}")
        for level in all_levels
    ]
    
    handles = level_patches

    if show_dep == True:
        chain_patches = [
            mlines.Line2D(
                [], [],
                color= chain_color_map[c],
                linewidth=1.5,
                marker=">",
                markersize=6,
                label=f"Chain {c}",
            )
            for c in all_chains if c in arrow_chain_set
        ]

        handles += chain_patches

    if show_legend:
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.65,
        )
        
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# Modify to suit your needs
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
        save_path           = f"uptime_{slurm_job_id}.pdf",
    )
