from typing import List

import pandas as pd


class MissingInput(Exception):
    ...


def validate_existing_columns(df: pd.DataFrame, required: List["str"]) -> bool:
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise MissingInput(
            f"The following columns are missing from the input: {missing}"
        )
    return True
