from typing import Union

import numpy as np
import pandas as pd
import scikitplot as skplt


def evaluate_model(
    true_label: Union[pd.Series, np.ndarray],
    pred_label,
    plot_confusion=True,
    confusion_title=None,
) -> dict:
    """Compute binary classification metrics and plot confusion matrix"""
    tp = ((true_label == 1) & (pred_label == 1)).sum()
    fp = ((true_label == 0) & (pred_label == 1)).sum()
    tn = ((true_label == 0) & (pred_label == 0)).sum()
    fn = ((true_label == 1) & (pred_label == 0)).sum()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    if plot_confusion:
        skplt.metrics.plot_confusion_matrix(
            true_label, pred_label, title=confusion_title
        )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "specificity": round(specificity, 3),
    }
