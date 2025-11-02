import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# --- Constants ---
OUTPUTS_DIR = Path("./outputs")
PLOTS_DIR = Path("./plots")

PLOT_DEFINITIONS = [
    {
        "title": "XGBoost: Strategy Comparison (IID)",
        "filename": "fig_1_xgboost_strategies_iid.png",
        "filters": {
            "partitioner-type": "uniform"
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"}
    },
    {
        "title": "XGBoost: Strategy Comparison (non-IID, α=0.4)",
        "filename": "fig_2_xgboost_strategies_noniid_alpha_0_4.png",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.4
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"}
    },
    {
        "title": "XGBoost: Strategy Comparison (non-IID, α=0.8)",
        "filename": "fig_3_xgboost_strategies_noniid_alpha_0_8.png",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.8
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"}
    },
]

# Load all results.json files and combine into a single DataFrame
def load_all_results(outputs_dir: Path) -> pd.DataFrame:
    all_data = []
    for json_path in outputs_dir.rglob("results.json"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            config = data.get("config", {})
            if "scenario_name" not in config:
                continue
            for round_data in data.get("rounds", []):
                row = {**config, **round_data}
                if 'client_evaluation' in row:
                    row.update({f"client_evaluation_{k}": v for k, v in row['client_evaluation'].items()})
                all_data.append(row)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Skipping corrupted or unreadable file: {json_path} ({e})")
    return pd.DataFrame(all_data)

def generate_standard_plot(df: pd.DataFrame, plot_def: dict):
    title = plot_def["title"]
    filename = plot_def["filename"]
    comparison_key = plot_def["comparison_key"]
    legend_map = plot_def.get("legend_map", {})

    if df.empty:
        print(f"Skipping plot '{title}' - no data found after filtering.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) 
    fig.suptitle(title, fontsize=26, y=1.0)

    plot_configs = [
        ("Client-Side Aggregated Accuracy", "client_evaluation_accuracy", "Accuracy", axes[0]),
        ("Client-Side Aggregated F1-Score", "client_evaluation_f1_score", "Macro F1-Score", axes[1]),
        ("Client-Side Aggregated AUC", "client_evaluation_auc", "AUC", axes[2])
    ]

    scenarios = sorted(df[comparison_key].unique())
    colors = plt.get_cmap('tab10').colors
    linestyles = ['-', '--', '-.', ':'] * (len(scenarios) // 4 + 1)

    for i, name in enumerate(scenarios):
        group = df[df[comparison_key] == name]
        label = legend_map.get(name, name)
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        plot_args = {'label': label, 'color': color, 'linestyle': linestyle, 'linewidth': 2.5}
        
        for _, col, _, ax in plot_configs:
            if col in group.columns and not group[col].isnull().all():
                agg = group.groupby("round")[col].agg(['mean', 'std']).fillna(0)
                ax.plot(agg.index, agg['mean'], **plot_args)
                ax.fill_between(agg.index, agg['mean'] - agg['std'], agg['mean'] + agg['std'], color=color, alpha=0.1)

    max_round = df["round"].max() if not df.empty else 15
    for ax_title, _, ylabel, ax in plot_configs:
        ax.set_title(ax_title, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel("Federated Learning Round", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-0.01, 1.05) 
        ax.set_xlim(-0.5, max_round + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto', steps=[1, 2, 5]))
        ax.tick_params(axis='both', which='major', labelsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    num_cols = min(4, len(scenarios))
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25) 

    fig.legend(handles, labels, title="Strategy", loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=num_cols, 
               fontsize=18, title_fontsize=20)
    
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_path = PLOTS_DIR / filename
    fig.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Generated plot: {plot_path}")

def generate_plot(df: pd.DataFrame, plot_def: dict):
    generate_standard_plot(df, plot_def)

if __name__ == "__main__":
    print("Starting analysis...")
    master_df = load_all_results(OUTPUTS_DIR)
    
    if not master_df.empty:
        for plot_def in PLOT_DEFINITIONS:
            filtered_df = master_df.copy()
            filters = plot_def.get("filters", {})
            for key, value in filters.items():
                if key in filtered_df.columns:
                    if isinstance(value, list):
                        filtered_df = filtered_df[filtered_df[key].isin(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[key] == value]
            
            generate_plot(df=filtered_df, plot_def=plot_def)
            
        print("\nAnalysis complete. Plots saved in 'plots' directory.")
    else:
        print("\nCould not find any 'results.json' files in the 'outputs' directory.")