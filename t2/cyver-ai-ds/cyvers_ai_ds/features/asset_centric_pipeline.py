"""Asset (sender) centric feature engineering pipeline"""

from typing import List, Optional, Tuple

import pandas as pd

from .sender import DEFAULT_ROLLING_FEATURES, compute_sender_ratios
from .sender_receiver import compute_sender_receiver_info
from .transaction import (
    DEFAULT_AGG_LIST,
    DEFAULT_CLASS_COUNTS,
    group_by_tx_hash,
)


class AssetCentricFEPipeline:
    """Asset centric feature engineering pipeline

    The execute method of this class with accept a dataframe and
    perform the complete asset centric feature engineering pipeline.
    This pipeline includes:
    1. Compute sender-receiver based statistical features.
    2. Aggregate features to transaction level (grouped by transaction hash).
    3. Convert feature values to relative quantities with respect to the
        per-asset rolling window aggregations.

    Attributes:
        sender_id_col: Name of column containing sender (asset) id.
        receiver_id_col: Name of column containing receiver id.
        timestamp_col: Name of column containing transaction timestamp.
        label_col: Name of column containing labels.
        amount_usd_col: Name of column containing transferred amount in USD.
        currency_col: Name of column containing the type of currency used
            (ETH or other token type).
        transaction_id_col: Name of column containing transaction hashes.
        tx_agg_list: Recipe for transaction hash aggregation step.
        tx_count_classes: Recipe for class counts during transaction
            level aggregation.
        rolling_features: List of features to use when computing ratios to
            asset level rolling means.
        rolling_window_size: Window size for asset rolling aggregations.
        rolling_window_size_unit: Time unit for asset rolling aggregations.
        rolling_ratio_cutoff: Cutoff for ratio features computed from
            rolling window aggregation on asset level.

    Returns:
        Tuple of (
            feature dataframe,
            asset time series dataframe
        )
    """

    def __init__(
        self,
        # general attributes
        sender_id_col: str = "sender_id",
        receiver_id_col: str = "receiver_id",
        timestamp_col: str = "block_time",
        label_col: str = "label",
        amount_usd_col: str = "amount_usd",
        currency_col: str = "currency",
        # transaction aggregation attributes
        transaction_id_col: str = "transaction_id",
        tx_agg_list: Optional[List[Tuple[str, List[str]]]] = DEFAULT_AGG_LIST,
        tx_count_classes: Optional[
            List[Tuple[str, List[str]]]
        ] = DEFAULT_CLASS_COUNTS,
        # ratio to rolling by asset attributes
        rolling_features: List[str] = DEFAULT_ROLLING_FEATURES,
        rolling_window_size: float = 30,
        rolling_window_size_unit: str = "d",
        rolling_ratio_cutoff: float = 15,
    ):
        self.sender_id_col = sender_id_col
        self.receiver_id_col = receiver_id_col
        self.timestamp_col = timestamp_col
        self.label_col = label_col
        self.amount_usd_col = amount_usd_col
        self.currency_col = currency_col
        self.transaction_id_col = transaction_id_col
        self.tx_agg_list = tx_agg_list
        self.tx_count_classes = tx_count_classes
        self.rolling_features = rolling_features
        self.rolling_window_size = rolling_window_size
        self.rolling_window_size_unit = rolling_window_size_unit
        self.rolling_ratio_cutoff = rolling_ratio_cutoff

    def execute(self, df: pd.DataFrame):

        # make sure data is sorted by time and add index
        # to make sure merge is valid
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        df["index"] = df.index

        init_df_size = df.shape[0]

        # 1. Compute features per sender-receiver combination
        df = df.merge(
            compute_sender_receiver_info(
                df,
                sender_id_col=self.sender_id_col,
                receiver_id_col=self.receiver_id_col,
                amount_usd_col=self.amount_usd_col,
                currency_col=self.currency_col,
                timestamp_col=self.timestamp_col,
                index_col="index",
            )
        )

        assert df.shape[0] == init_df_size, (
            "Data lost in sender_receiver feature engineering",
        )

        # 2. Aggregate to transaction level features
        n_transactions = df[self.transaction_id_col].nunique()
        df = group_by_tx_hash(
            df,
            tx_id_col=self.transaction_id_col,
            agg_list=self.tx_agg_list,
            count_classes=self.tx_count_classes,
            label_col=self.label_col,
        )
        assert df[self.transaction_id_col].nunique() == n_transactions, (
            "transactions lost in aggregation",
        )

        # 3. Normalize to asset rolling aggregation
        df, asset_df = compute_sender_ratios(
            df,
            rolling_features=self.rolling_features,
            sender_id_col=self.sender_id_col,
            window_size=self.rolling_window_size,
            window_size_unit=self.rolling_window_size_unit,
            timestamp_col=self.timestamp_col,
            ratio_cutoff=self.rolling_ratio_cutoff,
        )

        return df, asset_df
