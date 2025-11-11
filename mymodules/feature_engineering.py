import numpy as np
import pandas as pd

def _has(df, cols):  # 필요한 컬럼이 모두 있는지
    return all(c in df.columns for c in cols)

def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)

# -------- A 파생 --------
def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6

    # 0) Year-Month 단일축
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    # 1) 속도-정확도 트레이드오프
    if _has(feats, ["A1_rt_mean","A1_resp_rate"]):
        feats["A1_speed_acc_tradeoff"] = _safe_div(feats["A1_rt_mean"], feats["A1_resp_rate"], eps)
    if _has(feats, ["A2_rt_mean","A2_resp_rate"]):
        feats["A2_speed_acc_tradeoff"] = _safe_div(feats["A2_rt_mean"], feats["A2_resp_rate"], eps)
    if _has(feats, ["A4_rt_mean","A4_acc_rate"]):
        feats["A4_speed_acc_tradeoff"] = _safe_div(feats["A4_rt_mean"], feats["A4_acc_rate"], eps)

    # 2) RT 변동계수(CV)
    for k in ["A1","A2","A3","A4"]:
        m = f"{k}_rt_mean"
        s = f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    # 3) 조건 차이 절댓값(편향 크기)
    for name, base in [
        ("A1_rt_side_gap_abs",  "A1_rt_side_diff"),
        ("A1_rt_speed_gap_abs", "A1_rt_speed_diff"),
        ("A2_rt_cond1_gap_abs", "A2_rt_cond1_diff"),
        ("A2_rt_cond2_gap_abs", "A2_rt_cond2_diff"),
        ("A4_stroop_gap_abs",   "A4_stroop_diff"),
        ("A4_color_gap_abs",    "A4_rt_color_diff"),
    ]:
        if base in feats.columns:
            feats[name] = feats[base].abs()

    # 4) 정확도 패턴 심화
    if _has(feats, ["A3_valid_ratio","A3_invalid_ratio"]):
        feats["A3_valid_invalid_gap"] = feats["A3_valid_ratio"] - feats["A3_invalid_ratio"]
    if _has(feats, ["A3_correct_ratio","A3_invalid_ratio"]):
        feats["A3_correct_invalid_gap"] = feats["A3_correct_ratio"] - feats["A3_invalid_ratio"]
    if _has(feats, ["A5_acc_change","A5_acc_nonchange"]):
        feats["A5_change_nonchange_gap"] = feats["A5_acc_change"] - feats["A5_acc_nonchange"]

    # 5) 간단 메타 리스크 스코어(휴리스틱)
    parts = []
    if "A4_stroop_gap_abs" in feats: parts.append(0.30 * feats["A4_stroop_gap_abs"].fillna(0))
    if "A4_acc_rate" in feats:       parts.append(0.20 * (1 - feats["A4_acc_rate"].fillna(0)))
    if "A3_valid_invalid_gap" in feats:
        parts.append(0.20 * feats["A3_valid_invalid_gap"].fillna(0).abs())
    if "A1_rt_cv" in feats: parts.append(0.20 * feats["A1_rt_cv"].fillna(0))
    if "A2_rt_cv" in feats: parts.append(0.10 * feats["A2_rt_cv"].fillna(0))
    if parts:
        feats["RiskScore"] = sum(parts)

    # NaN/inf 정리
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

# -------- B 파생 --------
def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6

    # 0) Year-Month 단일축
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    # 1) 속도-정확도 트레이드오프 (B1~B5)
    # 반응속도를 정확도로 나눈다.
    for k, acc_col, rt_col in [
        ("B1", "B1_acc_task1", "B1_rt_mean"), # XXX ??? B1의 반응시간은 과제2의 데이터라메?
        ("B2", "B2_acc_task1", "B2_rt_mean"), # XXX ??? B2의 반응시간은 과제2의 데이터라메?
        ("B3", "B3_acc_rate",  "B3_rt_mean"),
        ("B4", "B4_acc_rate",  "B4_rt_mean"),
        ("B5", "B5_acc_rate",  "B5_rt_mean"),
    ]:
        if _has(feats, [rt_col, acc_col]):
            feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[rt_col], feats[acc_col], eps)

    # 2) RT 변동계수(CV)
    for k in ["B1","B2","B3","B4","B5"]:
        m = f"{k}_rt_mean"
        s = f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    # 3) 간단 메타 리스크 스코어(휴리스틱)
    parts = []
    for k in ["B4","B5"]:  # 주의집중/스트룹 유사 과제 가중
        if _has(feats, [f"{k}_rt_cv"]):
            parts.append(0.25 * feats[f"{k}_rt_cv"].fillna(0))
    for k in ["B3","B4","B5"]:
        acc = f"{k}_acc_rate" if k != "B1" and k != "B2" else None
        if k in ["B1","B2"]:
            acc = f"{k}_acc_task1"
        if acc in feats:
            parts.append(0.25 * (1 - feats[acc].fillna(0)))
    for k in ["B1","B2"]:
        tcol = f"{k}_speed_acc_tradeoff"
        if tcol in feats:
            parts.append(0.25 * feats[tcol].fillna(0))
    if parts:
        feats["RiskScore_B"] = sum(parts)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats