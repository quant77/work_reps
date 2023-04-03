"""Asset level feature engineering.

Feature engineering based on asset properties and transaction history.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from cyvers_ai_ds.data.validation import validate_existing_columns

from .timestamps import rolling_window_agg
from .transaction import DEFAULT_AGG_LIST as TX_AGG_LIST

# default list of features for rolling window aggregation by sender
# derived from the transaction level aggregation part
NON_FEATURE_COLS = ["transaction_id", "sender_id", "block_time", "label"]
TX_FEATURE_COLS = []
for x in TX_AGG_LIST:
    if x[0] not in NON_FEATURE_COLS:
        src = x[0]
        for agg in x[1]:
            TX_FEATURE_COLS.append(
                (src + "_tx_" + agg).replace("_tx_first", "")
            )

# additional feature - count of internal transactions
TX_FEATURE_COLS.append("internal_tx_cnt")


DEFAULT_ROLLING_FEATURES = TX_FEATURE_COLS

VALID_RATIO_TYPES = {
    "log_to_median",
    "log_to_mean",
    "diff_from_median_q_norm",
}


def compute_sender_ratios(
    df: pd.DataFrame,
    rolling_features: List[str] = DEFAULT_ROLLING_FEATURES,
    sender_id_col: str = "sender_id",
    window_size: float = 30,
    window_size_unit: str = "d",
    timestamp_col: str = "block_time",
    ratio_types: List[str] = ["log_to_median"],
    ratio_cutoff: float = 15,
    q_range: Tuple[float, float] = (20, 80),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ratio to sender current throughput of transactions.

    For each feature in the rolling feature list,
    this function will compute the rolling aggregations at the sender level
    and features that are derived from them (see detailed below under 'ratio_types').

    Args:
        df: Data containing transaction level features.
        rolling_features: List of features on which to compute the
            ratio to sender aggregation.
        sender_id_col: Column containing sender ids. Will be used
            to compute rolling aggregation per sender.
        window_size: Window size for rolling aggregation.
        window_size_unit: Time unit for window size aggregation.
            Should be understandable by pandas timedelta function.
        timestamp_col: Column containing transaction timestamps.
        ratio_type: Types of ratio features to compute.
            Options are:
                'log_to_median' - log(value / rolling median per asset),
                'log_to_mean' - log(value / rolling mean per asset),
                'diff_from_median_q_norm' - (value - rolling_median) /
                    (high quantile value - low quantile value)
        ratio_cutoff: Cutoff to use for the (non-logarithmic) ratio features.
            (Trivial computation can lead to infinities because
            the rolling std can be zero)
        q_rang: Low and high percentiles to be computed in a rolling aggregation
            per asset. Should be numbers in range 0-100.

    Returns:
        Tuple containing:
            - Original df with ratio features added.
            - Df with sender rolling aggregations.
    """

    req_cols = [sender_id_col, timestamp_col] + rolling_features
    validate_existing_columns(df, req_cols)

    for r in ratio_types:
        assert (
            r in VALID_RATIO_TYPES
        ), f"unrecognized ratio type: {r}, allowed types are {VALID_RATIO_TYPES}"

    # compute rolling aggregations per sender
    rolled_df = rolling_window_agg(
        df,
        cols_to_aggregate=rolling_features,
        partition_by=sender_id_col,
        timestamp_col=timestamp_col,
        window_size=window_size,
        window_size_unit=window_size_unit,
        aggs=["median", "mean", "std"],
        quantiles=[q_range[0] / 100, q_range[1] / 100],
    )
    df = df.sort_values([sender_id_col, timestamp_col]).reset_index(drop=True)
    rolled_df = rolled_df.sort_values(
        [sender_id_col, timestamp_col]
    ).reset_index(drop=True)
    assert (
        (
            df.loc[:, [sender_id_col, timestamp_col]]
            == rolled_df.loc[:, [sender_id_col, timestamp_col]]
        )
        .all()
        .all()
    ), "rolling output does not match input"

    # compute features relative to asset aggregations

    ratio_cols = []

    if "diff_from_median_q_norm" in ratio_types:
        for f in rolling_features:
            df[f + "_ratio"] = (df[f] - rolled_df[f + "_rolling_median"]) / (
                rolled_df[f"{f}_rolling_qt_{q_range[0]}"]
                - rolled_df[f"{f}_rolling_qt_{q_range[1]}"]
            )
            ratio_cols.append(f + "_ratio")

    if "log_to_median" in ratio_types:
        for f in rolling_features:
            df[f + "_log_to_median_ratio"] = np.log(
                df[f].replace(0, 1)
                / rolled_df[f + "_rolling_median"].replace(0, 1)
            )

    if "log_to_mean" in ratio_types:
        for f in rolling_features:
            df[f + "_log_to_mean_ratio"] = np.log(
                df[f].replace(0, 1)
                / rolled_df[f + "_rolling_mean"].replace(0, 1)
            )

    # use cutoffs for extreme ratios
    for c in ratio_cols:
        above_thr = df[c].abs() > ratio_cutoff
        df.loc[above_thr, c] = (
            df[c][above_thr] / df[c][above_thr].abs() * ratio_cutoff
        )
    df.loc[:, ratio_cols].replace(-np.inf, -ratio_cutoff, inplace=True)
    df.loc[:, ratio_cols].replace(np.inf, ratio_cutoff, inplace=True)

    return df, rolled_df
