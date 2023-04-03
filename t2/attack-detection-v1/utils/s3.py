"""Utilities for working with AWS s3."""

from io import BytesIO
from typing import Optional

import pandas as pd

GLOBAL_DATA_BUCKET = "cyvers-ai-datasets"
MODELS_BUCKET = "cyvers-ai-models"


def read_csv_from_s3(
    bucket, key: str, read_csv_args: Optional[dict] = None
) -> pd.DataFrame:
    """Read file from s3 as a pandas dataframe.

    Args:
        bucket: Boto3 bucket instance.
        key: File key
        read_csv_args: Keyword arguments for pandas read_csv method.
    """
    if read_csv_args is None:
        read_csv_args = {}
    resp = bucket.Object(key).get()
    return pd.read_csv(BytesIO(resp["Body"].read()), **read_csv_args)
