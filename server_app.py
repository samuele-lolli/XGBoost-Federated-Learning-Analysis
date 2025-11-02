import json
import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common import ConfigRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from task import create_run_dir, replace_keys
from strategy import CustomFedXgbBagging, CustomFedXgbCyclic

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Load the full experiment configuration from the JSON file.
    with open("current_run_config.json", "r") as f:
        config = json.load(f)

    num_rounds = config["num-server-rounds"]
    fraction_train = config["fraction-train"]
    fraction_evaluate = config["fraction-evaluate"]
    client_timeout = config["client-timeout"]
    train_method = config.get("train-method", "bagging")
    
    # Select and initialize the federated learning strategy based on the config.
    if train_method == "bagging":
        strategy = CustomFedXgbBagging(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
        print("Strategy selected: FedXgbBagging")
    elif train_method == "cyclic":
        strategy = CustomFedXgbCyclic(fraction_train=fraction_train, fraction_evaluate=fraction_evaluate)
        print("Strategy selected: FedXgbCyclic")
    else:
        raise ValueError(f"Unknown training method: '{train_method}'")
    
    # Configure the strategy with the output paths and run config for logging.
    save_path, run_dir_name = create_run_dir()
    strategy.set_save_path_and_run_dir(
        path=save_path,
        run_dir=run_dir_name,
        num_rounds_planned=num_rounds,
        config=config,
    )

    # Prepare the initial global model (empty for XGBoost).
    global_model_bytes = b""
    initial_arrays = ArrayRecord([np.frombuffer(global_model_bytes, dtype=np.uint8)])
    
    # Start the federated learning process using Flower's standard execution engine.
    print(f"Starting strategy.start() for {num_rounds} rounds...")
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        timeout=client_timeout,
        train_config=ConfigRecord(config),
        evaluate_config=ConfigRecord(config)
    )

    # Save the final global model to a file.
    print("\nFederated training complete.")
    params = replace_keys(unflatten_dict(config))["params"]
    bst = xgb.Booster(params=params)
    final_model_bytes = bytearray(result.arrays["0"].numpy().tobytes())
    if final_model_bytes:
        bst.load_model(final_model_bytes)
        print("Saving final model to disk...")
        bst.save_model(str(save_path / "final_model.json"))
    else:
        print("No final model to save.")