import numpy as np
import pandas as pd
import pytest

from cyvers_ai_ds.features.sender_receiver import compute_sender_receiver_info


@pytest.fixture
def snd_rcv_history():
    np.random.seed(1234)
    n1 = 100
    n2 = 300
    n3 = 150
    n = n1 + n2 + n3

    dates = pd.date_range("2020-01-01", "2020-06-01", freq="1h")

    df1 = pd.DataFrame()
    df1["block_time"] = np.random.choice(dates, size=n1)
    df1["sender_id"] = "sender1"
    df1["receiver_id"] = "receiver1"
    df1["amount_usd"] = np.random.normal(12, 3, size=n1)
    df1["currency"] = np.random.choice(["eth", "usdt", "duckling"], size=n1)

    df2 = pd.DataFrame()
    df2["block_time"] = np.random.choice(dates, size=n2)
    df2["sender_id"] = "sender1"
    df2["receiver_id"] = "receiver2"
    df2["amount_usd"] = np.random.normal(4, 1, size=n2)
    df2["currency"] = np.random.choice(["eth", "usdt"], size=n2)

    df3 = pd.DataFrame()
    df3["block_time"] = np.random.choice(dates, size=n3)
    df3["sender_id"] = "sender2"
    df3["receiver_id"] = "receiver2"
    df3["amount_usd"] = np.random.normal(1, 0.2, size=n3)
    df3["currency"] = "eth"

    df = (
        pd.concat([df1, df2, df3], axis=0)
        .sample(n, replace=False)
        .reset_index(drop=True)
    )
    df["index"] = df.index

    return df


def test_compute_sender_receiver_info(snd_rcv_history):
    snd_rcv_info = compute_sender_receiver_info(
        snd_rcv_history, index_col="index"
    )

    # Check that all records are in the output
    assert (
        snd_rcv_info.loc[:, ["index", "sender_id", "receiver_id"]]
        .sort_values("index")
        .values
        == snd_rcv_history.loc[:, ["index", "sender_id", "receiver_id"]].values
    ).all()

    # Check latest records in the output match the
    # expected maximum values
    g = snd_rcv_info.groupby(["sender_id", "receiver_id"])

    assert g["snd_rcv_mean_amt_usd"].last().reset_index().to_dict()[
        "snd_rcv_mean_amt_usd"
    ] == {
        0: 12.16423129211804,
        1: 3.98048758837073,
        2: 1.0211232175982718,
    }

    assert g["snd_rcv_tx_cnt"].last().reset_index().to_dict()[
        "snd_rcv_tx_cnt"
    ] == {0: 100, 1: 300, 2: 150}

    # check that the first records match minimal values
    assert g["snd_rcv_tx_cnt"].first().reset_index().to_dict()[
        "snd_rcv_tx_cnt"
    ] == {0: 1, 1: 1, 2: 1}
