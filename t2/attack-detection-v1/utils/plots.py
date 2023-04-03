from typing import Any, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def plot_case(
    df: pd.DataFrame, file_name: str, features: List[str], max_points: int = 5000
) -> Tuple[plt.Figure, Sequence[Any]]:
    """Plot timeline of transaction features for an attack case (file name).
    Real and predicted label for each transaction will be marked.

    Args:
        df: Data to use. Should include the columns:
            'file_name', 'block_time', 'label', 'pred_label'
            in addition to the features to be displayed.
        file_name: The filename of the case we want to plot.
        features: list of features to plot.
        max_points: Max number of points to show in each plot.
            (most recent points will be used)

    Returns:
        Tuple of (figure, sequence of matplotlib axes subplots)
    """
    if type(features) is str:
        features = [features]
    plot_data = df.loc[
        df["file_name"] == file_name, ["block_time", "label", "pred_label"] + features
    ]
    if plot_data.shape[0] > max_points:
        plot_data = plot_data.sort_values("block_time").tail(max_points)
    fig, axs = plt.subplots(ncols=1, nrows=len(features), figsize=[12, 10])
    fig.suptitle(file_name)
    for i in range(len(features)):
        ax = axs[i]
        feature = features[i]
        ax.set_title(feature)
        is_tn = (plot_data["label"] == 0) & (plot_data["pred_label"] == 0)
        is_tp = (plot_data["label"] == 1) & (plot_data["pred_label"] == 1)
        is_fp = (plot_data["label"] == 0) & (plot_data["pred_label"] == 1)
        is_fn = (plot_data["label"] == 1) & (plot_data["pred_label"] == 0)
        ax.scatter(
            plot_data["block_time"][is_tn],
            plot_data[feature][is_tn],
            color="green",
            label="tn",
        )
        ax.scatter(
            plot_data["block_time"][is_tp],
            plot_data[feature][is_tp],
            color="red",
            label="tp",
        )
        ax.scatter(
            plot_data["block_time"][is_fp],
            plot_data[feature][is_fp],
            marker="X",
            color="green",
            label="fp",
            edgecolors="red",
            linewidths=1,
        )
        ax.scatter(
            plot_data["block_time"][is_fn],
            plot_data[feature][is_fn],
            marker="X",
            color="red",
            edgecolors="green",
            label="fn",
            linewidths=1,
        )
        ax.legend()
    fig.subplots_adjust(hspace=0.4)
    return fig, axs
