# utils/metrics.py
from typing import Dict, List, Tuple
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np

def compute_basic_metrics(y_true: List[int], y_pred: List[int], labels: List[str]) -> Dict:
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }

def probs_to_preds(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)
