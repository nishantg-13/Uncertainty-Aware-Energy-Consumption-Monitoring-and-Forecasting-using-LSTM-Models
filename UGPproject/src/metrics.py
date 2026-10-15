from __future__ import annotations

import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(
            np.mean(np.abs(error) / np.maximum(np.abs(actual), 1e-3)) * 100
        ),
    }


def uncertainty_metrics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    covered = (actual >= lower) & (actual <= upper)
    return {
        "interval_coverage_percent": float(np.mean(covered) * 100),
        "mean_interval_width_kw": float(np.mean(upper - lower)),
    }


def binary_detection_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    true_positive = int(np.sum(actual & predicted))
    false_positive = int(np.sum(~actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    true_negative = int(np.sum(~actual & ~predicted))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (true_positive + true_negative) / max(1, len(actual))
    return {
        "accuracy_percent": float(accuracy * 100),
        "precision_percent": float(precision * 100),
        "recall_percent": float(recall * 100),
        "f1_percent": float(f1 * 100),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def masked_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {"mae_kw": float("nan"), "rmse_kw": float("nan"), "points": 0}
    return {
        **regression_metrics(np.asarray(actual)[mask], np.asarray(predicted)[mask]),
        "points": int(np.sum(mask)),
    }

