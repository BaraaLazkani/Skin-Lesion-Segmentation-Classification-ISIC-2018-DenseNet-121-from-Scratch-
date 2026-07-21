import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw)


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d


def load_artifacts(acfg) -> SimpleNamespace:
    def _require(path, label):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"{label} not found at: {p}\n"
                f"Make sure training has been run and logs have been saved."
            )
        return p

    history_path  = _require(acfg.metrics_csv, "metrics.csv")
    best_path     = _require(acfg.best_metrics_json, "best_metrics.json")
    final_path    = _require(acfg.final_metrics_json, "final_metrics.json")
    cm_path       = _require(acfg.best_confusion_npy, "best_confusion_matrix.npy")

    history = pd.read_csv(history_path)

    with open(best_path) as f:
        best = json.load(f)
    with open(final_path) as f:
        final = json.load(f)

    confusion = np.load(cm_path)

    return SimpleNamespace(history=history, best=best, final=final, confusion=confusion)
