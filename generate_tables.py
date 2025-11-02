import json
from pathlib import Path
import pandas as pd

OUTPUTS_DIR = Path("./outputs")
TABLES_DIR = Path("./tables")

# Table Definitions for XGBoost Experiment
TABLE_DEFINITIONS = [
    {
        "title": "Table 1: Performance Comparison in IID Scenario",
        "filename": "table_1_iid.md",
        "filters": {
            "partitioner-type": "uniform"
        },
        "index_col": "train-method",
    },
    {
        "title": "Table 2: Performance Comparison in Non-IID Scenario (α=0.4)",
        "filename": "table_2_non_iid_a04.md",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.4
        },
        "index_col": "train-method",
    },
    {
        "title": "Table 3: Performance Comparison in Non-IID Scenario (α=0.8)",
        "filename": "table_3_non_iid_a08.md",
        "filters": {
            "partitioner-type": "dirichlet",
            "dirichlet-alpha": 0.8
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
                
            record = {**config, **summary}
            all_summaries.append(record)
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"Skipping corrupted or unreadable file: {json_path} ({e})")
            
    if not all_summaries:
        return pd.DataFrame()
        
    return pd.DataFrame(all_summaries)

# Formats metric columns as 'mean ± std' strings
def format_metric(mean_series: pd.Series, std_series: pd.Series) -> pd.Series:
    return mean_series.round(4).astype(str) + " ± " + std_series.round(4).astype(str)

def generate_markdown_table(df: pd.DataFrame, title: str, index_col_name: str) -> str:
    df.index.name = index_col_name.replace('-', ' ').replace('_', ' ').title()
    markdown = f"### {title}\n\n"
    markdown += df.to_markdown()
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
        for key, value in table_def["filters"].items():
            filtered_df = filtered_df[filtered_df[key] == value]

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
            formatted_cols[clean_name] = format_metric(agg_df[(metric, 'mean')], agg_df[(metric, 'std')])
        
        final_df = pd.DataFrame(formatted_cols)
        
        # Ensure a consistent order of strategies in the table rows
        strategy_order = ["Fed-Bagging", "Fed-Cyclic"]
        if final_df.index.name == 'Train Method': # Check against the prettified name
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