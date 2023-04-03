"""Feature engineering tools based on groupby operations."""

from typing import List, Optional, Tuple, Union

import pandas as pd


def group_data(
    df: pd.DataFrame,
    key_cols: Union[str, List[str]],
    agg_list: Optional[List[Tuple[str, List[str]]]] = None,
    count_classes: Optional[List[Tuple[str, List[str]]]] = None,
    name_suffix: str = "",
    cnt_col: Optional[str] = None,
) -> pd.DataFrame:
    """Generate feature based on groupby operation.

    Args:
        df: Data to use.
        key_cols: Columns to group by.
        agg_list: List of columns and aggregations to apply to them.
            See example below.
        count_classes: List of columns on which to apply class count.
            See example below.
        name_suffix: suffix used in the
            feature names appearing in the output.
        cnt_col: If specified, a row count aggregation will
            be computed and this will be the name of the feature
            in the output.

    Returns:
        Dataframe with aggregated features

    Note:
        The output will not be of the same shape as the input!
        Multiple rows with the same value in key_cols will be merged
        together.

    Examples:
        1. The following an entry in agg_list:

        agg_list=[
            (col1, ['mean', 'std', ])
            ...
        ]

        will lead to the following columns in the output:
        col1_mean, col1_std

        2. The following entry in count_classes:
        [
            ('receiver_type', ['wallet', 'smart_contract'])
        ]
        Will lead to the following fields in the output with
        counts per group:
        receiver_type_wallet, receiver_type_smart_contract

    """
    if type(key_cols) is str:
        key_cols = [key_cols]
    g = df.groupby(key_cols)
    result = pd.DataFrame()
    if agg_list is None:
        agg_list = []


    for c, aggs in agg_list:
        for agg in aggs:
            result[c + name_suffix + "_" + agg] = g[c].agg(agg)

    #print('in group_data , 1 result.cols : ' , result.columns)        
    if count_classes is not None:
        dummy_df = df.loc[:, key_cols + [x[0] for x in count_classes]].copy()
        for c, classes in count_classes:
            for cls in classes:
                dummy_df[c + "_" + str(cls)] = (dummy_df[c] == cls).astype(int)
        g1 = dummy_df.groupby(key_cols)
        for c, classes in count_classes:
            for cls in classes:
                cls_name = str(cls)
                result[c + "_" + cls_name + name_suffix + "_cnt"] = g1[
                    c + "_" + cls_name
                ].sum()
    #print('in group_data , 2 result.cols : ' , result.columns)
    if cnt_col is not None:
        result[cnt_col] = g[key_cols].count()

    #print('in group_data , 3 result.cols : ' , result.columns)    
    return result.reset_index()
