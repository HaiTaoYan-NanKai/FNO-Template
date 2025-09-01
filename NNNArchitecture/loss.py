
# -*- coding: utf-8 -*-
import numpy as np

def compute_metrics(pred: np.ndarray, true: np.ndarray):
    assert pred.shape == true.shape
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"MSE": mse, "MAE": mae, "R2": r2}
