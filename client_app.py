import warnings
import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from task import load_data, replace_keys
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, log_loss

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()

def _local_boost(bst_input, num_local_round, train_dmatrix, train_method):
    """Performs local XGBoost model updating."""
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())
    
    # For bagging, return only the newly added trees. For cyclic, return the whole model.
    bst = (
        bst_input[bst_input.num_boosted_rounds() - num_local_round : bst_input.num_boosted_rounds()]
        if train_method == "bagging"
        else bst_input
    )
    return bst

@app.train()
def train(msg: Message, context: Context) -> Message:
    server_config = msg.content["config"]
    cfg = replace_keys(unflatten_dict(server_config))
    params = cfg["params"]
    
    train_dmatrix, _, num_train, _ = load_data(
        partitioner_type=cfg["partitioner_type"],
        partition_id=context.node_config["partition-id"],
        num_partitions=cfg["num_clients"],
        test_fraction=cfg["test_fraction"],
        seed=cfg["seed"],
        alpha=cfg.get("dirichlet_alpha", 0.0)
    )

    # Load global model from server and perform local training.
    if server_config["server-round"] == 1:
        bst = xgb.train(params, train_dmatrix, num_boost_round=cfg["local_epochs"])
    else:
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        bst.load_model(global_model)
        bst = _local_boost(bst, cfg["local_epochs"], train_dmatrix, cfg["train_method"])

    # Return the updated local model artifacts.
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)
    model_record = ArrayRecord([model_np])
    metric_record = MetricRecord({"num-examples": num_train})
    return Message(content=RecordDict({"arrays": model_record, "metrics": metric_record}), reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    server_config = msg.content["config"]
    cfg = replace_keys(unflatten_dict(server_config))
    params = cfg["params"]
    
    _, valid_dmatrix, _, num_val = load_data(
        partitioner_type=cfg["partitioner_type"],
        partition_id=context.node_config["partition-id"],
        num_partitions=cfg["num_clients"],
        test_fraction=cfg["test_fraction"],
        seed=cfg["seed"],
        alpha=cfg.get("dirichlet_alpha", 0.0)
    )

    if num_val == 0:
        return Message(content=RecordDict({
            "metrics": MetricRecord({"num-examples": 0}),
            "arrays": ArrayRecord([np.zeros((2, 2), dtype=np.int64)])
        }), reply_to=msg)

    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)

    y_true = valid_dmatrix.get_label()
    y_pred_prob = bst.predict(valid_dmatrix)
    y_pred_binary = (y_pred_prob > 0.5).astype(int)
    
    loss, auc, accuracy = 0.0, 0.0, 0.0
    
    if len(np.unique(y_true)) > 1:
        loss = log_loss(y_true, y_pred_prob, labels=[0, 1])
        auc = roc_auc_score(y_true, y_pred_prob)
        accuracy = accuracy_score(y_true, y_pred_binary)
    else:
        accuracy = accuracy_score(y_true, y_pred_binary)

    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])

    metrics = {
        "loss": loss,
        "auc": auc,
        "accuracy": accuracy,
        "num-examples": num_val,
    }
    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics), "arrays": ArrayRecord([cm.astype(np.int64)])}), 
        reply_to=msg
    )