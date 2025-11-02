import json
import time
from logging import INFO
from pathlib import Path
from typing import Optional, Any, Tuple, List, Dict
import numpy as np
from flwr.common import log
from flwr.app import Message, ArrayRecord, MetricRecord

# Mixin class to add custom metric aggregation and JSON logging to Flower strategies.
class StrategyMixin:
    num_rounds_planned: Optional[int] = None
    save_path: Path
    run_dir: str
    run_config: Dict
    _current_train_metrics: Dict[str, Any] = {}
    _current_eval_metrics: Dict[str, Any] = {}

    def set_save_path_and_run_dir(self, path: Path, run_dir: str, num_rounds_planned: int, config: Dict) -> None:
        self.save_path = path
        self.run_dir = run_dir
        self.num_rounds_planned = num_rounds_planned
        self.run_config = config

    def aggregate_train(
        self,
        server_round: int,
        replies: List[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        # Call the original aggregation logic from the base strategy (e.g., FedXgbBagging).
        aggregated_arrays, aggregated_metrics = super().aggregate_train(server_round, replies)
        self._current_train_metrics = aggregated_metrics if aggregated_metrics is not None else {}
        return aggregated_arrays, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: List[Message],
    ) -> Optional[MetricRecord]:
        if not replies:
            return None

        # Filter out any failed client replies.
        valid_replies = [msg for msg in replies if msg.content is not None and msg.content["metrics"].get("num-examples", 0) > 0]
        if not valid_replies:
            log(INFO, "No valid replies received from clients in this evaluation round.")
            return None
        
        num_examples_total = sum(msg.content["metrics"]["num-examples"] for msg in valid_replies)

        # Calculate weighted average for standard metrics.
        weighted_loss = sum(msg.content["metrics"]["loss"] * msg.content["metrics"]["num-examples"] for msg in valid_replies) / num_examples_total
        weighted_accuracy = sum(msg.content["metrics"]["accuracy"] * msg.content["metrics"]["num-examples"] for msg in valid_replies) / num_examples_total
        weighted_auc = sum(msg.content["metrics"]["auc"] * msg.content["metrics"]["num-examples"] for msg in valid_replies) / num_examples_total

        # Aggregate confusion matrices to calculate a robust, weighted F1-score.
        confusion_matrices = [msg.content["arrays"]["0"].numpy() for msg in valid_replies if "arrays" in msg.content]
        macro_f1_from_cm = 0.0
        if confusion_matrices:
            aggregated_cm = np.sum(confusion_matrices, axis=0)
            
            # Calculate TP, FP, FN from the aggregated confusion matrix.
            tp = np.diag(aggregated_cm)
            fp = np.sum(aggregated_cm, axis=0) - tp
            fn = np.sum(aggregated_cm, axis=1) - tp
            
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
            recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
            
            f1_per_class = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
            
            macro_f1_from_cm = np.mean(f1_per_class)

        self._current_eval_metrics = {
            "loss": weighted_loss,
            "accuracy": weighted_accuracy,
            "auc": weighted_auc,
            "f1_score": macro_f1_from_cm,
            "num-examples": num_examples_total,
        }

        # Save results to file at the end of each round.
        self.save_metrics_and_log(
            current_round=server_round,
            train_metrics=self._current_train_metrics,
            eval_metrics=self._current_eval_metrics
        )
        
        return self._current_eval_metrics

    def save_metrics_and_log(
        self, 
        current_round: int, 
        train_metrics: Dict[str, Any], 
        eval_metrics: Dict[str, Any]
    ) -> None:
        """Saves the metrics for the current round to a JSON file."""
        def round_metric(value: Any, decimals: int) -> Any:
            return round(float(value), decimals) if isinstance(value, (int, float, np.floating)) else value

        round_data = {
            "round": current_round,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training": {"num_examples": train_metrics.get("num-examples", 0)},
            "client_evaluation": {
                "loss": round_metric(eval_metrics.get("loss", 0), 6),
                "accuracy": round_metric(eval_metrics.get("accuracy", 0), 4),
                "f1_score": round_metric(eval_metrics.get("f1_score", 0), 4),
                "auc": round_metric(eval_metrics.get("auc", 0), 4),
                "num_examples": eval_metrics.get("num-examples", 0)
            }
        }
        
        results_file = self.save_path / "results.json"
        results_data = {"config": {}, "rounds": []}
        
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as fp:
                try:
                    results_data = json.load(fp)
                except json.JSONDecodeError:
                    log(INFO, "results.json is corrupted, creating a new one.")
                    
        if not results_data.get("config"):
            results_data["config"] = self.run_config
        
        rounds = results_data.get("rounds", [])
        existing_round_index = next((i for i, r in enumerate(rounds) if r.get("round") == current_round), -1)
        if existing_round_index != -1:
            rounds[existing_round_index] = round_data
        else:
            rounds.append(round_data)
        
        results_data["rounds"] = sorted(rounds, key=lambda x: x["round"])
        results_data["summary"] = self._calculate_summary_statistics(results_data["rounds"])
        
        try:
            with open(results_file, "w", encoding="utf-8") as fp:
                json.dump(results_data, fp, indent=2)
        except IOError as e:
            log(INFO, "Error saving results.json: %s", e)
        
        self._print_round_metrics(current_round, round_data)

    def _calculate_summary_statistics(self, rounds: List[Dict]) -> Dict:
        """Calculates summary statistics from all completed rounds."""
        if not rounds:
            return {}
        client_eval_rounds = [r.get("client_evaluation", {}) for r in rounds]
        return {
            "total_rounds_completed": len(rounds),
            "best_client_accuracy": max(r.get("accuracy", 0) for r in client_eval_rounds),
            "best_client_f1_score": max(r.get("f1_score", 0) for r in client_eval_rounds),
            "best_client_auc": max(r.get("auc", 0) for r in client_eval_rounds),
            "final_client_accuracy": client_eval_rounds[-1].get("accuracy", 0),
            "final_client_f1_score": client_eval_rounds[-1].get("f1_score", 0),
            "final_client_auc": client_eval_rounds[-1].get("auc", 0),
        }

    def _print_round_metrics(self, current_round: int, round_data: Dict) -> None:
        """Logs a summary of the current round's metrics to the console."""
        log(INFO, f"--- ROUND {current_round}/{self.num_rounds_planned} METRICS SUMMARY ---")
        if client_eval := round_data.get("client_evaluation"):
            log(INFO, "  [Client Eval Agg] Acc: %.4f, F1: %.4f, AUC: %.4f",
                client_eval.get("accuracy", 0), client_eval.get("f1_score", 0), 
                client_eval.get("auc", 0))
        log(INFO, "-------------------------------------------------")