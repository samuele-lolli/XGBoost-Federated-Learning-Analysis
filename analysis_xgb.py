import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import traceback 

OUTPUTS_DIR = Path("./outputs")
PLOTS_DIR = Path("./plots")

PLOT_DEFINITIONS = [
    # ==========================================================
    # --- PARTECIPAZIONE TOTALE (fraction_train = 1.0) ---
    # ==========================================================
    {
        "title": "XGBoost: Strategy Comparison (IID, Full Participation)",
        "filename": "fig_1_xgboost_strategies_iid_full.pdf", 
        "filters": {
            "partitioner-type": "uniform",
            "fraction-train": 1.0
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"}
    },
    {
        "title": "XGBoost: Strategy Comparison (non-IID, α=0.4, Full Participation)",
        "filename": "fig_2_xgboost_strategies_noniid_alpha_0_4_full.pdf",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.4,
            "fraction-train": 1.0 
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"}
    },
    {
        "title": "XGBoost: Strategy Comparison (non-IID, α=0.8, Full Participation)",
        "filename": "fig_3_xgboost_strategies_noniid_alpha_0_8_full.pdf", 
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.8,
            "fraction-train": 1.0 
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"}
    },
    
    # ==========================================================
    # --- PARTECIPAZIONE PARZIALE (fraction_train = 0.8) ---
    # ==========================================================
    {
        "title": "XGBoost: Strategy Comparison (non-IID, α=0.4, fraction=0.8)",
        "filename": "fig_5_xgboost_strategies_noniid_alpha_0_4_frac08.pdf",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.4,
            "fraction-train": 0.8
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"},
        "include_full_participation_baseline": True 
    },
    {
        "title": "XGBoost: Strategy Comparison (non-IID, α=0.8, fraction=0.8)",
        "filename": "fig_6_xgboost_strategies_noniid_alpha_0_8_frac08.pdf",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.8,
            "fraction-train": 0.8
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"},
        "include_full_participation_baseline": True 
    },

    # ==========================================================
    # --- PARTECIPAZIONE PARZIALE (fraction_train = 0.7) ---
    # ==========================================================
    {
        "title": "XGBoost: Strategy Comparison (non-IID, α=0.4, fraction=0.7)",
        "filename": "fig_8_xgboost_strategies_noniid_alpha_0_4_frac07.pdf",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.4,
            "fraction-train": 0.7
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"},
        "include_full_participation_baseline": True 
    },
    {
        "title": "XGBoost: Strategy Comparison (non-IID, α=0.8, fraction=0.7)",
        "filename": "fig_9_xgboost_strategies_noniid_alpha_0_8_frac07.pdf",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.8,
            "fraction-train": 0.7
        },
        "comparison_key": "train-method",
        "legend_map": {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"},
        "include_full_participation_baseline": True
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
            
            config.setdefault("fraction-train", 1.0)
            if config["fraction-train"] is None: 
                config["fraction-train"] = 1.0
            
            if config.get('partitioner-type') == 'uniform':
                 config.setdefault("dirichlet-alpha", -1.0) 
            elif 'dirichlet-alpha' not in config or config["dirichlet-alpha"] is None:
                 config["dirichlet-alpha"] = np.nan

            for round_data in data.get("rounds", []):
                row = {**config, **round_data}
                if 'client_evaluation' in row:
                    row.update({f"client_evaluation_{k}": v for k, v in row['client_evaluation'].items()})
                all_data.append(row)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Skipping corrupted or unreadable file: {json_path} ({e})")

    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)

    numeric_cols = ["round", "dirichlet-alpha", "fraction-train"] + \
                   [col for col in df.columns if "client_evaluation" in col and "num-examples" not in col]
    for col in numeric_cols:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'dirichlet-alpha' in df.columns and 'partitioner-type' in df.columns:
        is_iid_mask = df['partitioner-type'] == 'uniform' # XGB usa 'uniform' per IID
        df.loc[is_iid_mask, 'dirichlet-alpha'] = df.loc[is_iid_mask, 'dirichlet-alpha'].fillna(-1.0)

    return df

def generate_standard_plot(df: pd.DataFrame, plot_def: dict, baseline_df: pd.DataFrame = None):
    title = plot_def["title"]
    filename = plot_def["filename"]
    comparison_key = plot_def["comparison_key"]
    legend_map = plot_def.get("legend_map", {})

    if df.empty and (baseline_df is None or baseline_df.empty):
        print(f"Skipping plot '{title}' - no data found after filtering.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 20), sharex=True) 
    fig.suptitle(title, fontsize=26, y=0.98)

    plot_configs = [
        ("Client-Side Aggregated Accuracy", "client_evaluation_accuracy", "Accuracy", axes[0]),
        ("Client-Side Aggregated F1-Score", "client_evaluation_f1_score", "Macro F1-Score", axes[1]),
        ("Client-Side Aggregated AUC", "client_evaluation_auc", "AUC", axes[2])
    ]

    scenarios = sorted(df[comparison_key].unique())
    colors = plt.get_cmap('tab10').colors
    linestyles = ['-', '--', '-.', ':'] * (len(scenarios) // 4 + 1)

    handles = []; labels = [] 

    if baseline_df is not None and not baseline_df.empty:
        baseline_name = "Fed-Bagging (Full Participation)" 
        baseline_args = {'label': baseline_name, 'color': 'black', 'linestyle': '--', 'linewidth': 2.0}
        
        baseline_plotted = False
        for _, col, _, ax in plot_configs:
            try:
                if col in baseline_df.columns and not baseline_df[col].isnull().all():
                    agg = baseline_df.groupby("round")[col].agg(['mean', 'std']).fillna(0)
                    ax.plot(agg.index, agg['mean'], **baseline_args)
                    ax.fill_between(agg.index, agg['mean'] - agg['std'], agg['mean'] + agg['std'], color='black', alpha=0.1)
                    baseline_plotted = True
            except Exception as e:
                print(f"Error plotting baseline metric {col}: {e}")
        
        if baseline_plotted:
            line = plt.Line2D([0], [0], **baseline_args)
            handles.append(line)
            labels.append(baseline_name)

    for i, name in enumerate(scenarios):
        group = df[df[comparison_key] == name]
        label = legend_map.get(name, name)
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        plot_args = {'label': label, 'color': color, 'linestyle': linestyle, 'linewidth': 2.5}
        
        plotted_this_scenario = False
        for _, col, _, ax in plot_configs:
            try:
                if col in group.columns and not group[col].isnull().all():
                    agg = group.groupby("round")[col].agg(['mean', 'std']).fillna(0)
                    ax.plot(agg.index, agg['mean'], **plot_args)
                    ax.fill_between(agg.index, agg['mean'] - agg['std'], agg['mean'] + agg['std'], color=color, alpha=0.1)
                    plotted_this_scenario = True
            except Exception as e:
                print(f"Error plotting metric {col} for {name}: {e}")
                traceback.print_exc()

        if plotted_this_scenario:
            line = plt.Line2D([0], [0], **plot_args)
            handles.append(line)
            labels.append(label)

    max_round_data = max(
        df["round"].max() if not df.empty else 0, 
        baseline_df["round"].max() if baseline_df is not None and not baseline_df.empty else 0
    )
    max_round = int(max_round_data) if pd.notna(max_round_data) else 15

    xlabel_text = "Federated Learning Round"
    
    for ax_title, _, ylabel, ax in plot_configs:
        ax.set_title(ax_title, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-0.01, 1.05) 
        ax.set_xlim(-0.5, max_round + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto', steps=[1, 2, 5]))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel("") 
    
    axes[2].set_xlabel(xlabel_text, fontsize=16)

    num_cols = min(4, len(labels))
    
    fig.subplots_adjust(bottom=0.15, hspace=0.25) 

    fig.legend(handles, labels, title="Strategy", loc='lower center', 
               bbox_to_anchor=(0.5, 0.05), ncol=num_cols, 
               fontsize=18, title_fontsize=20)
    
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_path = PLOTS_DIR / filename
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated plot: {plot_path}")

def generate_plot(df: pd.DataFrame, plot_def: dict, baseline_df: pd.DataFrame = None):
    generate_standard_plot(df, plot_def, baseline_df=baseline_df)

if __name__ == "__main__":
    print("Starting analysis...")
    master_df = load_all_results(OUTPUTS_DIR)
    
    if not master_df.empty:
        required_cols = {'fraction-train': float, 'dirichlet-alpha': float, 'partitioner-type': str}
        default_values = {'fraction-train': 1.0, 'dirichlet-alpha': np.nan, 'partitioner-type': 'unknown'}
        for col, dtype in required_cols.items():
             if col not in master_df.columns: master_df[col] = default_values[col]
             if dtype == str: master_df[col] = master_df[col].fillna(default_values[col]).astype(str)
             elif dtype == float: master_df[col] = pd.to_numeric(master_df[col].fillna(default_values[col]), errors='coerce')
        
        if 'dirichlet-alpha' in master_df.columns:
             is_iid_mask = master_df['partitioner-type'] == 'uniform'
             master_df.loc[is_iid_mask, 'dirichlet-alpha'] = master_df.loc[is_iid_mask, 'dirichlet-alpha'].fillna(-1.0)

        for plot_def in PLOT_DEFINITIONS:
            filtered_df = master_df.copy()
            filters = plot_def.get("filters", {})
            baseline_data = None 
            
            for key, value in filters.items():
                if key in filtered_df.columns:
                    if isinstance(value, list):
                        filter_list = []
                        col_dtype = filtered_df[key].dtype
                        is_numeric_col = pd.api.types.is_numeric_dtype(col_dtype) and not pd.api.types.is_bool_dtype(col_dtype)
                        for v in value:
                            if v is None or v == "":
                                if is_numeric_col: filter_list.append(np.nan)
                                else: filter_list.append('none')
                            elif is_numeric_col: filter_list.append(pd.to_numeric(v, errors='coerce'))
                            else: filter_list.append(str(v))
                        
                        has_nan = any(pd.isna(v) for v in filter_list)
                        actual_values = [v for v in filter_list if pd.notna(v)]
                        mask = filtered_df[key].isin(actual_values)
                        if has_nan: mask = mask | filtered_df[key].isna()
                        filtered_df = filtered_df[mask]
                    else: 
                        filter_val = value
                        if filter_val is None or filter_val == "":
                            if key in ["fraction-train", "dirichlet-alpha"]: filter_val = np.nan
                            else: filter_val = 'none'

                        if pd.isna(filter_val) and key in ["dirichlet-alpha"]:
                            filtered_df = filtered_df[filtered_df[key].isna()]
                        elif isinstance(filter_val, (int, float)) and key in ["fraction-train", "dirichlet-alpha"]:
                            numeric_col = pd.to_numeric(filtered_df[key], errors='coerce')
                            if isinstance(filter_val, float):
                                filtered_df = filtered_df[np.isclose(numeric_col, filter_val, equal_nan=True)]
                            else:
                                filtered_df = filtered_df[numeric_col == filter_val]
                        else: 
                            filtered_df = filtered_df[filtered_df[key].astype(str) == str(filter_val)]
            
            include_full_participation_baseline_flag = plot_def.get("include_full_participation_baseline", False)

            if include_full_participation_baseline_flag:
                print(f"   Finding corresponding FULL PARTICIPATION baseline (Fed-Bagging, frac=1.0) data for: {plot_def['title']}")
                
                baseline_filters = {
                    "train-method": "bagging",
                    "fraction-train": 1.0 
                }
                
                key_filters_to_copy = ["partitioner-type", "dirichlet-alpha"]
                for key in key_filters_to_copy:
                    if key in filters:
                        baseline_filters[key] = filters[key]
                        
                print(f"   Baseline filters: {baseline_filters}")

                baseline_data = master_df.copy()
                
                for key, value in baseline_filters.items():
                    if key not in baseline_data.columns: continue
                    
                    filter_val = value
                    if isinstance(filter_val, (int, float)):
                        numeric_col = pd.to_numeric(baseline_data[key], errors='coerce')
                        if isinstance(filter_val, float): 
                            baseline_data = baseline_data[np.isclose(numeric_col, filter_val, equal_nan=True)]
                        else:
                            baseline_data = baseline_data[numeric_col == filter_val]
                    else:
                         baseline_data = baseline_data[baseline_data[key].astype(str) == str(filter_val)]

                if baseline_data.empty:
                     print("Full Participation Baseline (Fed-Bagging) data not found.")
            
            effective_baseline_df = baseline_data if include_full_participation_baseline_flag and baseline_data is not None and not baseline_data.empty else None

            generate_plot(df=filtered_df, plot_def=plot_def, baseline_df=effective_baseline_df)
            
        print("\nAnalysis complete. Plots saved in 'plots' directory.")
    else:
        print("\nCould not find any 'results.json' files in the 'outputs' directory.")