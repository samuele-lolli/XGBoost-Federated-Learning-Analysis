import json
from pathlib import Path
import pandas as pd
import numpy as np 
import traceback 

OUTPUTS_DIR = Path("./outputs")
TABLES_DIR = Path("./tables")

TABLE_DEFINITIONS = [
    # --- Partecipazione Totale (fraction=1.0) ---
    {
        "title": "Table 1: Performance Comparison in IID Scenario (Full Participation)",
        "filename": "table_1_iid_full.md",
        "filters": {
            "partitioner-type": "uniform",
            "fraction-train": 1.0
        },
        "index_col": "train-method",
    },
    {
        "title": "Table 2: Performance Comparison in Non-IID Scenario (α=0.4, Full Participation)",
        "filename": "table_2_non_iid_a04_full.md",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.4,
            "fraction-train": 1.0
        },
        "index_col": "train-method",
    },
    {
        "title": "Table 3: Performance Comparison in Non-IID Scenario (α=0.8, Full Participation)",
        "filename": "table_3_non_iid_a08_full.md",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.8,
            "fraction-train": 1.0
        },
        "index_col": "train-method",
    },
    # --- Partecipazione Parziale (fraction=0.8) ---
    {
        "title": "Table 4: Performance Comparison in IID Scenario (Partial Participation, fraction=0.8)",
        "filename": "table_4_iid_frac08.md",
        "filters": {
            "partitioner-type": "uniform",
            "fraction-train": 0.8
        },
        "index_col": "train-method",
    },
    {
        "title": "Table 5: Performance Comparison in Non-IID Scenario (α=0.4, Partial Participation, fraction=0.8)",
        "filename": "table_5_non_iid_a04_frac08.md",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.4,
            "fraction-train": 0.8
        },
        "index_col": "train-method",
    },
    {
        "title": "Table 6: Performance Comparison in Non-IID Scenario (α=0.8, Partial Participation, fraction=0.8)",
        "filename": "table_6_non_iid_a08_frac08.md",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.8,
            "fraction-train": 0.8
        },
        "index_col": "train-method",
    },
    # --- Partecipazione Parziale (fraction=0.7) ---
    {
        "title": "Table 7: Performance Comparison in IID Scenario (Partial Participation, fraction=0.7)",
        "filename": "table_7_iid_frac07.md",
        "filters": {
            "partitioner-type": "uniform",
            "fraction-train": 0.7
        },
        "index_col": "train-method",
    },
    {
        "title": "Table 8: Performance Comparison in Non-IID Scenario (α=0.4, Partial Participation, fraction=0.7)",
        "filename": "table_8_non_iid_a04_frac07.md",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.4,
            "fraction-train": 0.7
        },
        "index_col": "train-method",
    },
    {
        "title": "Table 9: Performance Comparison in Non-IID Scenario (α=0.8, Partial Participation, fraction=0.7)",
        "filename": "table_9_non_iid_a08_frac07.md",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.8,
            "fraction-train": 0.7
        },
        "index_col": "train-method",
    },
]

# Loads all summary data from results.json files into a single DataFrame
def load_summary_data(outputs_dir: Path) -> pd.DataFrame:
    all_summaries = []
    for json_path in outputs_dir.rglob("results.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            config = data.get("config", {})
            summary = data.get("summary", {})
            
            if "scenario_name" not in config or not summary:
                continue
            
            config.setdefault("fraction-train", 1.0)
            if config["fraction-train"] is None: 
                config["fraction-train"] = 1.0
            
            if config.get('partitioner-type') == 'uniform':
                 config.setdefault("dirichlet-alpha", -1.0) 
            elif 'dirichlet-alpha' not in config or config["dirichlet-alpha"] is None:
                 config["dirichlet-alpha"] = np.nan

            record = {**config, **summary}
            all_summaries.append(record)
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"Skipping corrupted or unreadable file: {json_path} ({e})")
            
    if not all_summaries:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_summaries)

    print("Performing data cleaning and type conversion...")
    metric_cols = [col for col in df.columns if col.startswith("final_client_")]
    numeric_cols = ["fraction-train", "dirichlet-alpha"] + metric_cols
    for col in numeric_cols:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce')

    string_cols = ["scenario_name", "train-method", "partitioner-type"]
    for col in string_cols:
         if col in df.columns: 
             df[col] = df[col].fillna('none').astype(str)
         else: 
             df[col] = 'none'

    if 'dirichlet-alpha' in df.columns and 'partitioner-type' in df.columns:
         is_iid_mask = df['partitioner-type'] == 'uniform'
         df.loc[is_iid_mask, 'dirichlet-alpha'] = df.loc[is_iid_mask, 'dirichlet-alpha'].fillna(-1.0)
    
    return df

# Formats metric columns as 'mean ± std' strings
def format_metric(mean_series: pd.Series, std_series: pd.Series, decimals=3) -> pd.Series: # Modificato decimals
    """Formats two series (mean, std) into a single 'mean ± std' string series."""
    std_series_filled = std_series.fillna(0)
    formatted = []
    for mean_val, std_val in zip(mean_series, std_series_filled):
         if pd.isna(mean_val):
              formatted.append("N/A")
         else:
              formatted.append(f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}")
    return pd.Series(formatted, index=mean_series.index)


def generate_markdown_table(df: pd.DataFrame, title: str, index_col_name: str) -> str:
    clean_index_name = index_col_name.replace('-', ' ').replace('_', ' ').title()
    if isinstance(df.index, pd.MultiIndex):
         df.index.names = [n.replace('-', ' ').replace('_', ' ').title() for n in df.index.names]
    elif not df.empty:
         df.index.name = clean_index_name

    markdown = f"### {title}\n\n"
    if df.empty:
        markdown += "*No data available for this configuration.*\n\n"
    else:
        ordered_cols = []
        possible_cols_order = ["Accuracy", "F1 Score", "Auc"]
        for col in possible_cols_order:
             if col in df.columns: ordered_cols.append(col)
        ordered_cols += sorted([col for col in df.columns if col not in ordered_cols])
        df_ordered = df[ordered_cols]
        markdown += df_ordered.to_markdown() 
        markdown += "\n\n"
    return markdown

def main():
    print("Starting table generation for XGBoost project...")
    master_df = load_summary_data(OUTPUTS_DIR)

    if master_df.empty:
        print("\nCould not find any 'results.json' files with summary data.")
        return

    TABLES_DIR.mkdir(exist_ok=True)
    
    # Standardize the strategy names for consistent labeling
    name_map = {"bagging": "Fed-Bagging", "cyclic": "Fed-Cyclic"}
    master_df['train-method'] = master_df['train-method'].replace(name_map)

    for table_def in TABLE_DEFINITIONS:
        print(f"Generating: {table_def['title']}")
        
        filtered_df = master_df.copy()
        
        try:
            filters = table_def.get('filters', {})
            for key, value in filters.items():
                if key not in filtered_df.columns:
                    print(f" Warning: Filter key '{key}' not found. Skipping filter.")
                    continue

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
        except Exception as e:
            print(f" Error during filtering for '{table_def['title']}': {e}")
            traceback.print_exc()
            continue

        if filtered_df.empty:
            print(f"  -> No data found for this table. Skipping.")
            continue
            
        # Metrics to aggregate for the XGBoost project
        metrics_to_agg = [
            "final_client_accuracy", "final_client_f1_score", "final_client_auc"
        ]
            
        # Group by the defined index column (e.g., 'train-method')
        agg_df = filtered_df.groupby(table_def["index_col"])[metrics_to_agg].agg(['mean', 'std'])

        # Format the aggregated columns into 'mean ± std' strings
        formatted_cols = {}
        for metric in metrics_to_agg:
            # Prettify column names
            clean_name = metric.replace("final_client_", "").replace("_", " ").title()
            if "Auc" in clean_name: clean_name = "AUC" # Caso speciale per AUC
            
            mean_col = (metric, 'mean')
            std_col = (metric, 'std')
            if mean_col in agg_df.columns:
                 std_series = agg_df[std_col] if std_col in agg_df.columns else pd.Series([0.0]*len(agg_df), index=agg_df.index)
                 formatted_cols[clean_name] = format_metric(agg_df[mean_col], std_series)
            else:
                 formatted_cols[clean_name] = "N/A"
        
        final_df = pd.DataFrame(formatted_cols)
        
        # Ensure a consistent order of strategies in the table rows
        strategy_order = ["Fed-Bagging", "Fed-Cyclic"]
        # Pulisce il nome dell'indice per il confronto
        clean_index_name = table_def["index_col"].replace('-', ' ').replace('_', ' ').title()
        
        if final_df.index.name == clean_index_name:
            current_strategies = [s for s in strategy_order if s in final_df.index]
            final_df = final_df.reindex(current_strategies)
            
        # Generate and print the Markdown table
        markdown_output = generate_markdown_table(
            df=final_df, title=table_def['title'], index_col_name=table_def['index_col']
        )
        print(markdown_output)
        
        # Save the table to a file
        table_path = TABLES_DIR / table_def['filename']
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        print(f"  -> Saved to {table_path}")

    print("\nTable generation complete. Files are in the 'tables' directory.")

if __name__ == "__main__":
    main()