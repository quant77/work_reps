import numpy as np
import pandas as pd
import pytest
from pytest import approx

from cyvers_ai_ds.features.group import group_data


@pytest.fixture
def mixed_data():
    np.random.seed(1234)

    n1 = 100
    n2 = 300
    n3 = 150
    n = n1 + n2 + n3

    df1 = pd.DataFrame()
    df1["f1"] = np.random.normal(1, 0.5, size=n1)
    df1["f2"] = np.random.choice([1, 2, 3], size=n1)
    df1["group"] = 1

    df2 = pd.DataFrame()
    df2["f1"] = np.random.normal(3, 0.2, size=n2)
    df2["f2"] = np.random.choice([1, 2], size=n2)
    df2["group"] = 2

    df3 = pd.DataFrame()
    df3["f1"] = np.random.normal(6, 0.5, size=n3)
    df3["f2"] = np.random.choice([2, 3, 4, 5, 6], size=n3)
    df3["group"] = 3

    df = (
        pd.concat([df1, df2, df3], axis=0)
        .sample(n, replace=False)
        .reset_index(drop=True)
    )
    return df


def test_count_classes(mixed_data):
    grouped = group_data(
        mixed_data,
        key_cols="group",
        count_classes=[("f2", [1, 2, 3, 4, 5, 6])],
    )
    assert grouped.to_dict() == {
        "group": {0: 1, 1: 2, 2: 3},
        "f2_1_cnt": {0: 34, 1: 148, 2: 0},
        "f2_2_cnt": {0: 41, 1: 152, 2: 36},
        "f2_3_cnt": {0: 25, 1: 0, 2: 34},
        "f2_4_cnt": {0: 0, 1: 0, 2: 17},
        "f2_5_cnt": {0: 0, 1: 0, 2: 41},
        "f2_6_cnt": {0: 0, 1: 0, 2: 22},
    }


def test_agg_list(mixed_data):
    grouped = group_data(
        mixed_data,
        key_cols="group",
        agg_list=[("f1", ["mean", "min", "max"]), ("f2", ["nunique"])],
    )
    assert grouped.to_dict() == {
        "group": {0: 1, 1: 2, 2: 3},
        "f1_mean": {
            0: 1.0175561415627181,
            1: 2.988837168446793,
            2: 5.9533218867071245,
        },
        "f1_min": {
            0: -0.7817583303123676,
            1: 2.3716052411336306,
            2: 4.801249385878165,
        },
        "f1_max": {
            0: 2.1954802577315164,
            1: 3.598218696191403,
            2: 7.533588390108906,
        },
        "f2_nunique": {0: 3, 1: 2, 2: 5},
    }


def test_count_col(mixed_data):
    grouped = group_data(mixed_data, key_cols="group", cnt_col="record_count")
    assert grouped.to_dict() == {
        "group": {0: 1, 1: 2, 2: 3},
        "record_count": {0: 100, 1: 300, 2: 150},
    }


def test_name_suffix(mixed_data):
    grouped = group_data(
        mixed_data,
        key_cols="group",
        agg_list=[("f1", ["mean", "std"])],
        name_suffix="_grp",
    )
    assert list(grouped.columns) == ["group", "f1_grp_mean", "f1_grp_std"]
