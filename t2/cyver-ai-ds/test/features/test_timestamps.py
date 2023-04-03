import numpy as np
import pandas as pd
import pytest
from pytest import approx

from cyvers_ai_ds.features.timestamps import rolling_window_agg


@pytest.fixture
def timeseries_data():
    np.random.seed(1234)
    df = pd.DataFrame()
    df["dates"] = pd.date_range("2020-01-01", "2020-06-01", freq="1h")
    df["f1"] = np.random.normal(size=df.shape[0])
    df["f2"] = (df["dates"] - df["dates"][0]).dt.total_seconds()
    df["partition"] = (df.index.astype(float) / (df.shape[0] / 4)).astype(int)
    return df


def test_rolling_window_no_partition(timeseries_data):
    rolled = rolling_window_agg(
        timeseries_data, ["f1", "f2"], timestamp_col="dates"
    )
    assert rolled["f1_rolling_mean"].mean() == approx(0.0306, 0.01)
    assert rolled["f1_rolling_std"].mean() == approx(0.9872, 0.01)
    assert rolled["f2_rolling_median"].mean() == approx(5399882.10468, 0.01)
    assert rolled["f2_rolling_qt_20"].mean() == approx(4699971.3674, 0.01)


def test_rolling_window_with_partition(timeseries_data):
    rolled = rolling_window_agg(
        timeseries_data,
        ["f1", "f2"],
        timestamp_col="dates",
        aggs=["mean", "max", "min"],
        partition_by="partition",
    )
    # monotonic increasing
    assert (rolled["f2_rolling_max"] == timeseries_data["f2"]).all()

    assert rolled["f2_rolling_mean"].mean() == approx(5782928.418, 0.01)

    # check first record of group 1
    first_group1_idx = rolled.index[rolled["partition"] == 1][0]
    assert (
        rolled.loc[
            first_group1_idx,
            ["f1_rolling_mean", "f1_rolling_max", "f1_rolling_min"],
        ]
        == timeseries_data.loc[first_group1_idx, "f1"]
    ).all()


def test_rolling_window_quantiles_only(timeseries_data):
    rolled = rolling_window_agg(
        timeseries_data,
        ["f1", "f2"],
        timestamp_col="dates",
        aggs=[],
        quantiles=[0.1, 0.3, 0.6],
        partition_by="partition",
    )
    assert list(rolled.columns) == [
        "partition",
        "dates",
        "f1_rolling_qt_10",
        "f1_rolling_qt_30",
        "f1_rolling_qt_60",
        "f2_rolling_qt_10",
        "f2_rolling_qt_30",
        "f2_rolling_qt_60",
    ]
