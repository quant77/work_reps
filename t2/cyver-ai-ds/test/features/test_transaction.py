from cyvers_ai_ds.features.transaction import group_by_tx_hash

from .test_group import mixed_data


def test_group_by_tx_hash(mixed_data):
    result = group_by_tx_hash(
        mixed_data,
        tx_id_col="group",
        agg_list=[("f1", ["mean", "min", "max"]), ("f2", ["nunique"])],
        count_classes=[("f2", [1, 2, 3, 4, 5, 6])],
    )
    assert list(result.columns) == [
        "group",
        "f1_tx_mean",
        "f1_tx_min",
        "f1_tx_max",
        "f2_tx_nunique",
        "f2_1_tx_cnt",
        "f2_2_tx_cnt",
        "f2_3_tx_cnt",
        "f2_4_tx_cnt",
        "f2_5_tx_cnt",
        "f2_6_tx_cnt",
        "internal_tx_cnt",
    ]
