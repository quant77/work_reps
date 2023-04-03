from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def find_bin_breaks(
    s: pd.Series,
    n_bins: int = 5,
    remove_outliers: bool = True,
    outlier_sigma: float = 3,
    random_state: int = 1234,
) -> List[float]:
    """Compute breaks for binning a numerical feature.
    Uses KMeans to compute bin breaks.

    Args:
        s: Pandas series with numerical data.
        n_bins: Number of bins.
        remove_outlier: Whether to apply outlier removel before
            computing the bin breaks.
        outlier_sigma: Number of standard deviations to keep
            during outlier removal.
        random_state: Random state to be used in KMeans

    Returns:
        List of bin breaks values.
    """
    if remove_outliers:
        non_outlier = ((s - s.mean()) / s.std()).abs() < outlier_sigma
        s1 = s[non_outlier]
    else:
        s1 = s
    n_bins = min(n_bins, int(s1.nunique()))
    kmeans = KMeans(n_clusters=n_bins, n_init="auto", random_state=random_state)
    kmeans.fit(s1.values.reshape(-1, 1))
    centers = np.sort(kmeans.cluster_centers_[:, 0])
    breaks = list((centers[1:] + centers[:-1]) / 2)
    return breaks


def bin_numeric_feature(s: pd.Series, breaks: List[float]) -> pd.DataFrame:
    """Bin a numerical feature and convert it to a
        set of binary features.

    Args:
        s: Pandas series containing data.
        breaks: Breaks used to convert the numerical to bins.

    Returns:
        A dataframe with binary data
    """
    feature = s.name
    n = s.shape[0]
    n_breaks = len(breaks)
    extended_breaks = [-np.inf] + list(breaks) + [np.inf]
    result = np.zeros((n, n_breaks + 2))
    for i in range(n_breaks + 1):
        result[
            (s.values < extended_breaks[i + 1]) & (s.values >= extended_breaks[i]), i
        ] = 1
    result[s.isna().values, n_breaks + 1] = 1
    col_names = [feature + f"_bin{i}" for i in range(n_breaks + 1)] + [feature + "_nan"]
    result = pd.DataFrame(result, columns=col_names)
    return result
