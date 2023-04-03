"""Timestamp based feature engineering.

Should contain code for rolling aggregations, resampling, differences etc.
"""

from typing import List, Optional

import pandas as pd

from cyvers_ai_ds.data.validation import validate_existing_columns


def rolling_window_agg(
    df: pd.DataFrame,
    cols_to_aggregate: List[str],
    partition_by: Optional[str] = None,
    timestamp_col: str = "block_time",
    window_size: float = 30,
    window_size_unit: str = "d",
    aggs: List[str] = ["median", "mean", "std"],
    quantiles: List[float] = [0.2, 0.8],
) -> pd.DataFrame:
    """Compute rolling window aggregations.

    Args:
        df: Data to aggregate.
        cols_to_aggregate: List of columns to apply aggregations on.
        partition_by: Column to use for partitioning the data.
        timestamp_col: Column that holds the timestamps for the
            rolling aggregations.
        window_size: Window size for aggregation.
        window_size_unit: Time unit for window size.
        aggs: List of aggregations to apply.
        quantiles: List of quantiles to compute (should be numbers in range 0-1).
    """
    

    #print('in rolling_window_agg')
    
    req_cols = cols_to_aggregate + [timestamp_col]
    #print('req_cols : ' , req_cols)
    result_cols = [timestamp_col]
    #print('result_cols : ', result_cols)
    partitioning = partition_by is not None
    if partitioning:
        req_cols.append(partition_by)
        result_cols.append(partition_by)

    validate_existing_columns(df, req_cols)

    df = df.sort_values(timestamp_col).reset_index(drop=True)

    if partitioning:
        g = df.groupby(partition_by)
    else:
        g = df

    window = pd.Timedelta(window_size, unit=window_size_unit)
    rolling = g.rolling(on=timestamp_col, window=window)
    agg_out = []
    #print('for c in cols_to_aggregate : ' , cols_to_aggregate )
    for c in cols_to_aggregate:
        #print('for agg in aggs: : ' , aggs )
        for agg in aggs:
            s = rolling[c].agg(agg)
            s.name = c + "_rolling_" + agg
            agg_out.append(s)
        for q in quantiles:
            s = rolling[c].quantile(q)
            s.name = c + "_rolling_qt_" + str(round(q * 100))
            agg_out.append(s)
    if partitioning:
        result = pd.concat(agg_out, axis=1).reset_index()
    else:
        result = pd.concat([g.loc[:, result_cols]] + agg_out, axis=1)
    return result
