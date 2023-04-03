"""V1 model - classification based on amount and time statistics normalized to the asset rolling median."""

from typing import List, Optional, Tuple

import pandas as pd

from cyvers_ai_ds.data.validation import validate_existing_columns

from .base import CyversAiModel

REQ_INPUT_FIELDS = [
    "block_time",
    "transaction_id",
    "sender_id",
    "receiver_id",
    "amount_usd",
    "currency",
]

MODEL_FEATURES = [
    "snd_rcv_amt_usd_sum_tx_mean_log_to_median_ratio",
    "snd_rcv_mean_amt_usd_tx_median_log_to_median_ratio",
    "amount_usd_tx_sum_log_to_median_ratio",
    "snd_rcv_life_time_sec_tx_min_log_to_median_ratio",
    "snd_rcv_mean_time_diff_sec_tx_sum_log_to_median_ratio",
    "snd_rcv_tx_cnt_tx_sum_log_to_median_ratio",
    "snd_rcv_time_diff_sec_tx_mean_log_to_median_ratio",
    "snd_rcv_tkn_type_cnt_tx_mean_log_to_median_ratio",
]


class TimeAmountClassifier(CyversAiModel):
    """Classification based on amount and time statistics normalized to
    the asset rolling median."""

    def __init__(
        self, model_artifacts: dict, prob_thr: Optional[float] = None
    ):
        """
        Args:
            model_artifact: Dictionary containing 'model' and 'preprocessor' objects.
            prob_thr: Probability threshold used for assigning predicted label.
        """
        super().__init__(
            model_artifacts=model_artifacts, required_input=REQ_INPUT_FIELDS
        )
        self.prob_thr = prob_thr

    def preprocess(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.preprocessor.execute(data)

    def infer(self, data: pd.DataFrame) -> pd.DataFrame:

        validate_existing_columns(data, REQ_INPUT_FIELDS)

        tx_df, _ = self.preprocess(data)
        tx_df.fillna(0, inplace=True)

        result = pd.DataFrame()
        result["transaction_id"] = tx_df["transaction_id"]
        result["pred_prob"] = self.model.predict_proba(
            tx_df.loc[:, MODEL_FEATURES]
        )[:, 1]
        if self.prob_thr is not None:
            result["pred_label"] = (
                result["pred_prob"] > self.prob_thr
            ).astype(int)
        else:
            result["pred_label"] = self.model.predict(
                tx_df.loc[:, MODEL_FEATURES]
            )
        return result
