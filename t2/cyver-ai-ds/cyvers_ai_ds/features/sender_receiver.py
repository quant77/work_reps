"""Features based on sender-receiver combination."""
from typing import Optional

import pandas as pd


def compute_sender_receiver_info(
    df: pd.DataFrame,
    sender_id_col: str = "sender_id",
    receiver_id_col: str = "receiver_id",
    amount_usd_col: str = "amount_usd",
    currency_col: str = "currency",
    timestamp_col: str = "block_time",
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute information about the history between a sender and receiver

    Args:
        df: Data to use.
        sender_id_col: Column used to identify senders.
        receiver_id_col: Column used to identify receivers.
        amount_usd_col: Column containing transfer amount in usd.
        currency_col: Column containing the currency used (token type or eth).
        timestamp_col: Column containing the block time of the interaction.
        index_col: If specified this column will be carried over to the result

    Returns:
        A dataframe with computed features in the shape of the original dataframe.
        The following features are aggregations for sender-receiver combinations:
        snd_rcv_tkn_type_cnt - number of currency types used.
        snd_rcv_tx_cnt - count of all transactions (internal and regular).
        snd_rcv_amt_usd_sum - sum of all transfers in usd
        snd_rcv_mean_amt_usd - mean of all transfers in usd
        snd_rcv_time_diff_sec - time difference from previous interaction in seconds.
        snd_rcv_life_time_sec - time from first interaction to current one in seconds.
        snd_rcv_mean_time_diff_sec - mean time difference between interactions.

    Note:
        The output can be merged with the original dataframe.

    """
    df = (
        df.copy()
        .sort_values([sender_id_col, receiver_id_col, timestamp_col])
        .reset_index(drop=True)
    )
    g = df.groupby([sender_id_col, receiver_id_col])
    agg_df = pd.DataFrame()
    agg_df["snd_rcv_min_time"] = g[timestamp_col].min()
    result_initial_cols = [sender_id_col, receiver_id_col, timestamp_col]
    if index_col is not None:
        result_initial_cols.append(index_col)
    token_type_count = (
        df.loc[:, [sender_id_col, receiver_id_col, currency_col]]
        .drop_duplicates()
        .groupby([sender_id_col, receiver_id_col])[currency_col]
        .cumcount()
        + 1
    )
    result = df.loc[:, result_initial_cols].copy()
    result["snd_rcv_tkn_type_cnt"] = token_type_count.reindex(df.index).ffill()
    result["snd_rcv_tx_cnt"] = g[timestamp_col].cumcount() + 1
    result["snd_rcv_amt_usd_sum"] = g[amount_usd_col].cumsum()
    result["snd_rcv_mean_amt_usd"] = (
        result["snd_rcv_amt_usd_sum"] / result["snd_rcv_tx_cnt"]
    )
    result["snd_rcv_time_diff_sec"] = (
        g[timestamp_col].diff().dt.total_seconds()
    )
    result = result.merge(agg_df.reset_index())
    result["snd_rcv_life_time_sec"] = (
        result["block_time"] - result["snd_rcv_min_time"]
    ).dt.total_seconds()
    result["snd_rcv_mean_time_diff_sec"] = result["snd_rcv_life_time_sec"] / (
        result["snd_rcv_tx_cnt"] - 1
    )
    return result
