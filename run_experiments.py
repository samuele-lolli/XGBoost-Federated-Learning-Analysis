import subprocess
import json
from typing import Dict, Any
import time
from run_configuration import scenarios

# The federation profile to use from pyproject.toml
FEDERATION_PROFILE = "local-simulation"

# Number of times each experimental scenario is repeated for statistical significance.
NUM_REPETITIONS = 15

# Total number of clients in the simulation pool.
# This must match the 'num-supernodes' in pyproject.toml.
NUM_CLIENTS = 8

# Base configuration with shared hyperparameters for all experiments.
BASE_CONFIG: Dict[str, Any] = {
    "num-server-rounds": 15,
    "fraction-train": 1.0,
    "fraction-evaluate": 1.0,
    "centralised-eval": False,
    "test-fraction": 0.2,
    "seed": 42,
    "local-epochs": 1,
    "client-timeout": 3600,
    # XGBoost base parameters
    "params.objective": "binary:logistic",
    "params.eta": 0.05,
    "params.max_depth": 4,
    "params.eval_metric": "auc",
    "params.nthread": 16,
    "params.num_parallel_tree": 1,
    "params.subsample": 0.8,
    "params.colsample_bytree": 0.8,
    "params.tree_method": "hist",
    "params.scale_pos_weight": 3.1527,
}

def run_experiment(config: Dict[str, Any]) -> bool:
    scenario_name = config.get("scenario_name", "Unnamed")
    print("-" * 50)
    print(f"RUNNING: {scenario_name} (Seed: {config['seed']})")
    print("-" * 50)

    config_path = "current_run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    command = ["flwr", "run", ".", FEDERATION_PROFILE, "--run-config", config_path]
    
    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as e:
        print(f"Error during experiment '{scenario_name}': {e}")
        return False
    finally:
        # Force Ray shutdown to clean up resources between runs.
        print("Forcing Ray shutdown to clean up resources...")
        subprocess.run(["ray", "stop", "--force"])
        print("Ray shutdown complete.")
    return True

if __name__ == "__main__":
    total_runs = len(scenarios) * NUM_REPETITIONS
    run_counter = 0

    print(f"Starting execution of {len(scenarios)} scenarios, {NUM_REPETITIONS} repetitions each. Total runs: {total_runs}")
    
    # Iterate through each defined scenario.
    for scenario_definition in scenarios:
        scenario_name = scenario_definition.get("scenario_name", "Unnamed Scenario")
        print(f"\n===== SCENARIO: {scenario_name} =====")

        # Repeat the scenario for NUM_REPETITIONS.
        for i in range(NUM_REPETITIONS):
            run_counter += 1
            exp_config = BASE_CONFIG.copy()
            exp_config.update(scenario_definition)
            
            # Add the total number of clients to the run configuration.
            exp_config["num-clients"] = NUM_CLIENTS
            
            # Increment the seed for each repetition to ensure statistical variety.
            current_seed = BASE_CONFIG["seed"] + i
            exp_config["seed"] = current_seed
            exp_config["scenario_name"] = scenario_name

            print(f"\n--- Running repetition {i+1}/{NUM_REPETITIONS} (Total: {run_counter}/{total_runs}) ---")
            
            if not run_experiment(exp_config):
                print("Script interrupted due to an error or user intervention.")
                exit()
            
            print(f"Repetition {i+1} finished. Waiting 5 seconds before the next...")
            time.sleep(5)
        
    print("\nAll experiments completed successfully.")