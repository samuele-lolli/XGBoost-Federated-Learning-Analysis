from typing import Any, Dict, List

# Defines the different experimental scenarios to be run.
# Each dictionary represents a unique experiment configuration that overrides BASE_CONFIG.
scenarios: List[Dict[str, Any]] = [
    {
        "scenario_name": "XGBoost_Bagging_IID",
        "train-method": "bagging",
        "partitioner-type": "uniform",
    },
    {
        "scenario_name": "XGBoost_Bagging_nonIID_a0_4",
        "train-method": "bagging",
        "partitioner-type": "dirichlet",
        "dirichlet-alpha": 0.4,
    },
    {
        "scenario_name": "XGBoost_Bagging_nonIID_a0_8",
        "train-method": "bagging",
        "partitioner-type": "dirichlet",
        "dirichlet-alpha": 0.8,
    },
    {
        "scenario_name": "XGBoost_Cyclic_IID",
        "train-method": "cyclic",
        "partitioner-type": "uniform",
    },
    {
        "scenario_name": "XGBoost_Cyclic_nonIID_a0_4",
        "train-method": "cyclic",
        "partitioner-type": "dirichlet",
        "dirichlet-alpha": 0.4,
    },
    {
        "scenario_name": "XGBoost_Cyclic_nonIID_a0_8",
        "train-method": "cyclic",
        "partitioner-type": "dirichlet",
        "dirichlet-alpha": 0.8,
    },
]