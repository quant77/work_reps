"""Transaction based features.

Features based on the transaction details.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from cyvers_ai_ds.data.validation import validate_existing_columns

from .group import group_data

#### Default settings for aggregation by transaction hash

# default list of aggregations
DEFAULT_AGGS = ["min", "max", "median", "mean", "std", "sum"]

# columns  to apply default aggregations on
DEFAULT_AGG_COLS = [
    "snd_rcv_tkn_type_cnt",
    "snd_rcv_tx_cnt",
    "snd_rcv_amt_usd_sum",
    "snd_rcv_mean_amt_usd",
    "snd_rcv_time_diff_sec",
    "snd_rcv_life_time_sec",
    "snd_rcv_mean_time_diff_sec",
    "amount_usd",
]

# columns from which to keep first value per transaction
DEFAULT_KEEP_FIRST_COLS = [
    "sender_id",
    "block_time",
    "gas_price",
    "gas_limit",
    "gas_used",
]

# columns from which to count unique values
DEFAULT_COUNT_UNIQUE_COLS = ["receiver_type"]

# complete default column and aggregation list
DEFAULT_AGG_LIST = (
    [(c, ["first"]) for c in DEFAULT_KEEP_FIRST_COLS]
    + [(c, DEFAULT_AGGS) for c in DEFAULT_AGG_COLS]
    + [(c, ["nunique"]) for c in DEFAULT_COUNT_UNIQUE_COLS]
)

# default columns and classes to sum up one hot encoding
DEFAULT_CLASS_COUNTS = [
    ("receiver_type", ["wallet", "smart_contract", "dex", "token"])
]


def group_by_tx_hash(
    df: pd.DataFrame,
    tx_id_col: str = "transaction_id",
    agg_list: Optional[List[Tuple[str, List[str]]]] = DEFAULT_AGG_LIST,
    count_classes: Optional[
        List[Tuple[str, List[str]]]
    ] = DEFAULT_CLASS_COUNTS,
    label_col: Optional[str] = "label",
) -> pd.DataFrame:
    """Aggregate data by transaction hash.

    Args:
        df: Data to process.
        tx_id_col: Transaction id column (used as key in groupby operation).
        agg_list: List of columns and aggregation to be applied.
        class_count: Columns and classes to sum up one hot encoding.
        label_col: Column containing labels. If it exists in the data
            the maximum label per transaction will be carried over to the result.

    Note:
        See more details on arguments in `cyvers_ai_ds.features.group.group_data`
    """
    req_cols = (
        [tx_id_col] + [x[0] for x in agg_list] + [x[0] for x in count_classes]
    )
    validate_existing_columns(df, req_cols)
    # carry over labels if they are present in the data
    if label_col in df.columns:
        agg_list += [(label_col, ["max"])]
    result = group_data(
        df,
        tx_id_col,
        agg_list=agg_list,
        count_classes=count_classes,
        name_suffix="_tx",
        cnt_col="internal_tx_cnt",
    )
    # first aggregations are not really aggregates so we can keep the original column name
    result.columns = result.columns.str.replace("_tx_first", "", regex=False)
    result.columns = result.columns.str.replace(
        label_col + "_tx_max", label_col
    )
    # replace infs in the data to nan
    result.replace([np.inf, -np.inf], np.nan, inplace=True)

    return result
