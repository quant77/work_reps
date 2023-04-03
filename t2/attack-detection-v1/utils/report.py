from typing import Tuple

import pandas as pd


def report_by_sections(result_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute aggregate performance metrics by section.

    Note:
        Expected columns in results_df:
        'label', 'pred_label', 'file_name', 'exp_part' (e.g., train or validation),
        'exploit_type' (e.g., 'Access Control')
    """
    true_label = result_df["label"]
    pred_label = result_df["pred_label"]
    result_df["tp"] = (true_label == 1) & (pred_label == 1)
    result_df["fp"] = (true_label == 0) & (pred_label == 1)
    result_df["tn"] = (true_label == 0) & (pred_label == 0)
    result_df["fn"] = (true_label == 1) & (pred_label == 0)

    # group by filename
    file_g = result_df.groupby("file_name")
    file_report = pd.DataFrame()
    file_report["exp_part"] = file_g["exp_part"].first()
    file_report["exploit_type"] = file_g["exploit_type"].first()
    file_report["sngl_mlt"] = file_g["sngl_mlt"].first()
    file_report["tp"] = file_g["tp"].sum()
    file_report["fp"] = file_g["fp"].sum()
    file_report["tn"] = file_g["tn"].sum()
    file_report["fn"] = file_g["fn"].sum()
    file_report["catch_any"] = (file_report["tp"] > 0).astype(int)
    file_report = file_report.reset_index()

    # group by exploit_type
    exploit_g = file_report.groupby(["exploit_type", "exp_part"])
    exploit_report = pd.DataFrame()
    exploit_report["files"] = exploit_g["file_name"].nunique()
    exploit_report["catch_any"] = exploit_g["catch_any"].mean()
    exploit_report["tp"] = exploit_g["tp"].sum()
    exploit_report["fp"] = exploit_g["fp"].sum()
    exploit_report["tn"] = exploit_g["tn"].sum()
    exploit_report["fn"] = exploit_g["fn"].sum()
    exploit_report = exploit_report.reset_index()

    exploit_report["recall"] = exploit_report["tp"] / (
        exploit_report["tp"] + exploit_report["fn"]
    )
    exploit_report["precision"] = exploit_report["tp"] / (
        exploit_report["tp"] + exploit_report["fp"]
    )

    # group by single/multiple attack
    sng_mlt_g = file_report.groupby(["sngl_mlt", "exp_part"])
    sng_mlt_report = pd.DataFrame()
    sng_mlt_report["files"] = sng_mlt_g["file_name"].nunique()
    sng_mlt_report["catch_any"] = sng_mlt_g["catch_any"].mean()
    sng_mlt_report["tp"] = sng_mlt_g["tp"].sum()
    sng_mlt_report["fp"] = sng_mlt_g["fp"].sum()
    sng_mlt_report["tn"] = sng_mlt_g["tn"].sum()
    sng_mlt_report["fn"] = sng_mlt_g["fn"].sum()
    sng_mlt_report = sng_mlt_report.reset_index()

    sng_mlt_report["recall"] = sng_mlt_report["tp"] / (
        sng_mlt_report["tp"] + sng_mlt_report["fn"]
    )
    sng_mlt_report["precision"] = sng_mlt_report["tp"] / (
        sng_mlt_report["tp"] + sng_mlt_report["fp"]
    )




    return file_report, exploit_report, sng_mlt_report
