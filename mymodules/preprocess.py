import numpy as np
import pandas as pd
from tqdm import tqdm

# =======================
# 학습 때 사용한 전처리 유틸 (그대로)
# =======================
tqdm.pandas()

"""
주어진 연령 문자열을 숫자로 변환합니다.
입력 예: "30a" -> 30, "30b" 등(끝 문자가 "a"가 아니면) -> base+5.
유효하지 않거나 결측이면 np.nan을 반환합니다.

Args:
    val: 연령을 나타내는 값(문자열 등)

Returns:
    float or int: 변환된 연령 또는 np.nan
"""
def convert_age(val):
    if pd.isna(val): return np.nan
    try:
        base = int(str(val)[:-1])
        return base if str(val)[-1] == "a" else base + 5
    except:
        return np.nan

"""
TestDate 값을 연도와 월로 분리합니다.
정수형 YYYYMM 형태를 예상하며, 실패 시 (np.nan, np.nan)을 반환합니다.

Args:
    val: 연도와 월이 결합된 값(예: 202301)

Returns:
    tuple: (year, month) 또는 (np.nan, np.nan)
"""
def split_testdate(val):
    try:
        v = int(val)
        return v // 100, v % 100
    except:
        return np.nan, np.nan

"""
콤마로 구분된 수열 문자열 시리즈에 대해 각 행의 평균을 계산합니다.
빈 문자열 또는 결측은 np.nan으로 처리합니다. tqdm 진행바를 사용합니다.

Args:
    series: 문자열 시리즈 (예: "1.0,2.0,3.0")

Returns:
    pd.Series: 각 행의 평균값 (인덱스 보존)
"""
def seq_mean(series):
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )

"""
콤마로 구분된 수열 문자열 시리즈에 대해 각 행의 표준편차(std)를 계산합니다.
빈 문자열 또는 결측은 np.nan으로 처리합니다. tqdm 진행바를 사용합니다.

Args:
    series: 문자열 시리즈 (예: "1.0,2.0,3.0")

Returns:
    pd.Series: 각 행의 표준편차 (인덱스 보존)
"""
def seq_std(series):
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )

"""
콤마로 구분된 응답 시퀀스에서 특정 값(target)의 비율을 계산합니다.
빈 문자열 또는 결측이면 np.nan을 반환합니다.

Args:
    series: 문자열 시리즈 (예: "1,0,1,1")
    target: 비율을 계산할 값 (문자열, 기본 "1")

Returns:
    pd.Series: 각 행에 대한 target 비율 (인덱스 보존)
"""
def seq_rate(series, target="1"):
    return series.fillna("").progress_apply(
        lambda x: str(x).split(",").count(target) / len(x.split(",")) if x else np.nan
    )

"""
두 개의 콤마 구분 시리즈(cond_series, val_series)를 같은 열 단위로 분해하여,
cond 값이 mask_val인 위치의 val 값들만 평균내어 반환합니다.
원소가 없거나 모두 마스킹되지 않으면 np.nan 처리합니다.

Args:
    cond_series: 조건을 담은 문자열 시리즈 (콤마 분리)
    val_series: 값을 담은 문자열 시리즈 (콤마 분리)
    mask_val: 마스크로 사용할 값 (숫자)

Returns:
    pd.Series: 각 행별로 마스크된 값들의 평균 (인덱스 보존)
"""
def masked_mean_from_csv_series(cond_series, val_series, mask_val):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)
    mask = (cond_arr == mask_val)
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts==0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

"""
두 개의 콤마 구분 시리즈(cond_series, val_series)를 같은 열 단위로 분해하여,
cond 값이 mask_set에 포함되는 위치의 val 값들만 평균내어 반환합니다.
원소가 없거나 해당 값이 없으면 np.nan 처리합니다.

Args:
    cond_series: 조건을 담은 문자열 시리즈 (콤마 분리)
    val_series: 값을 담은 문자열 시리즈 (콤마 분리)
    mask_set: 마스크로 사용할 값들의 집합 (예: {1,2,3})

Returns:
    pd.Series: 각 행별로 마스크된 값들의 평균 (인덱스 보존)
"""
def masked_mean_in_set_series(cond_series, val_series, mask_set):
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df  = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr  = val_df.to_numpy(dtype=float)
    mask = np.isin(cond_arr, list(mask_set))
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

"""
A 검사용 원본 DataFrame을 받아 여러 요약 특징(feature)을 생성하고 시퀀스 컬럼을 제거하여 반환합니다.
생성되는 주요 특징: Age_num, Year, Month 및 A1~A5 관련 반응률, 반응시간 평균/표준편차, 조건별 평균 등.
무한대 값을 np.nan으로 치환합니다.

Args:
    train_A: 원본 A 검사 DataFrame

Returns:
    pd.DataFrame: 파생특징이 추가되고 시퀀스 컬럼이 제거된 DataFrame
"""
def preprocess_A(train_A: pd.DataFrame) -> pd.DataFrame:
    df = train_A.copy()
    print("Step 1: Age, TestDate 파생...")
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    print("Step 2: A1 feature 생성...")
    feats["A1_resp_rate"] = seq_rate(df["A1-3"], "1")
    feats["A1_rt_mean"]   = seq_mean(df["A1-4"])
    feats["A1_rt_std"]    = seq_std(df["A1-4"])
    feats["A1_rt_left"]   = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 1)
    feats["A1_rt_right"]  = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 2)
    feats["A1_rt_side_diff"] = feats["A1_rt_left"] - feats["A1_rt_right"]
    feats["A1_rt_slow"]   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 1)
    feats["A1_rt_fast"]   = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 3)
    feats["A1_rt_speed_diff"] = feats["A1_rt_slow"] - feats["A1_rt_fast"]

    print("Step 3: A2 feature 생성...")
    feats["A2_resp_rate"] = seq_rate(df["A2-3"], "1")
    feats["A2_rt_mean"]   = seq_mean(df["A2-4"])
    feats["A2_rt_std"]    = seq_std(df["A2-4"])
    feats["A2_rt_cond1_diff"] = masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 1) - \
                                masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 3)
    feats["A2_rt_cond2_diff"] = masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 1) - \
                                masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 3)

    print("Step 4: A3 feature 생성...")
    s = df["A3-5"].fillna("")
    total   = s.apply(lambda x: len(x.split(",")) if x else 0)
    valid   = s.apply(lambda x: sum(v in {"1","2"} for v in x.split(",")) if x else 0)
    invalid = s.apply(lambda x: sum(v in {"3","4"} for v in x.split(",")) if x else 0)
    correct = s.apply(lambda x: sum(v in {"1","3"} for v in x.split(",")) if x else 0)
    feats["A3_valid_ratio"]   = (valid / total).replace([np.inf,-np.inf], np.nan)
    feats["A3_invalid_ratio"] = (invalid / total).replace([np.inf,-np.inf], np.nan)
    feats["A3_correct_ratio"] = (correct / total).replace([np.inf,-np.inf], np.nan)

    feats["A3_resp2_rate"] = seq_rate(df["A3-6"], "1")
    feats["A3_rt_mean"]    = seq_mean(df["A3-7"])
    feats["A3_rt_std"]     = seq_std(df["A3-7"])
    feats["A3_rt_size_diff"] = masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 1) - \
                               masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 2)
    feats["A3_rt_side_diff"] = masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 1) - \
                               masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 2)

    print("Step 5: A4 feature 생성...")
    feats["A4_acc_rate"]   = seq_rate(df["A4-3"], "1")
    feats["A4_resp2_rate"] = seq_rate(df["A4-4"], "1")
    feats["A4_rt_mean"]    = seq_mean(df["A4-5"])
    feats["A4_rt_std"]     = seq_std(df["A4-5"])
    feats["A4_stroop_diff"] = masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 2) - \
                              masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1)
    feats["A4_rt_color_diff"] = masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 1) - \
                                masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 2)

    print("Step 6: A5 feature 생성...")
    feats["A5_acc_rate"]   = seq_rate(df["A5-2"], "1")
    feats["A5_resp2_rate"] = seq_rate(df["A5-3"], "1")
    feats["A5_acc_nonchange"] = masked_mean_from_csv_series(df["A5-1"], df["A5-2"], 1)
    feats["A5_acc_change"]    = masked_mean_in_set_series(df["A5-1"], df["A5-2"], {2,3,4})

    print("Step 7: 시퀀스 컬럼 drop & concat...")
    seq_cols = [
        "A1-1","A1-2","A1-3","A1-4",
        "A2-1","A2-2","A2-3","A2-4",
        "A3-1","A3-2","A3-3","A3-4","A3-5","A3-6","A3-7",
        "A4-1","A4-2","A4-3","A4-4","A4-5",
        "A5-1","A5-2","A5-3"
    ]
    print("A 검사 데이터 전처리 완료")
    out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

"""
B 검사용 원본 DataFrame을 받아 여러 요약 특징(feature)을 생성하고 시퀀스 컬럼을 제거하여 반환합니다.
생성되는 주요 특징: Age_num, Year, Month 및 B1~B8 관련 정확도, 반응시간 평균/표준편차 등.
무한대 값을 np.nan으로 치환합니다.

Args:
    train_B: 원본 B 검사 DataFrame

Returns:
    pd.DataFrame: 파생특징이 추가되고 시퀀스 컬럼이 제거된 DataFrame
"""
def preprocess_B(train_B: pd.DataFrame) -> pd.DataFrame:
    df = train_B.copy()
    print("Step 1: Age, TestDate 파생...")
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    print("Step 2: B1 feature 생성...")
    feats["B1_acc_task1"] = seq_rate(df["B1-1"], "1")
    feats["B1_rt_mean"]   = seq_mean(df["B1-2"])
    feats["B1_rt_std"]    = seq_std(df["B1-2"])
    feats["B1_acc_task2"] = seq_rate(df["B1-3"], "1")

    print("Step 3: B2 feature 생성...")
    feats["B2_acc_task1"] = seq_rate(df["B2-1"], "1")
    feats["B2_rt_mean"]   = seq_mean(df["B2-2"])
    feats["B2_rt_std"]    = seq_std(df["B2-2"])
    feats["B2_acc_task2"] = seq_rate(df["B2-3"], "1")

    print("Step 4: B3 feature 생성...")
    feats["B3_acc_rate"] = seq_rate(df["B3-1"], "1")
    feats["B3_rt_mean"]  = seq_mean(df["B3-2"])
    feats["B3_rt_std"]   = seq_std(df["B3-2"])

    print("Step 5: B4 feature 생성...")
    feats["B4_acc_rate"] = seq_rate(df["B4-1"], "1")
    feats["B4_rt_mean"]  = seq_mean(df["B4-2"])
    feats["B4_rt_std"]   = seq_std(df["B4-2"])

    print("Step 6: B5 feature 생성...")
    feats["B5_acc_rate"] = seq_rate(df["B5-1"], "1")
    feats["B5_rt_mean"]  = seq_mean(df["B5-2"])
    feats["B5_rt_std"]   = seq_std(df["B5-2"])

    print("Step 7: B6~B8 feature 생성...")
    feats["B6_acc_rate"] = seq_rate(df["B6"], "1")
    feats["B7_acc_rate"] = seq_rate(df["B7"], "1")
    feats["B8_acc_rate"] = seq_rate(df["B8"], "1")

    print("Step 8: 시퀀스 컬럼 drop & concat...")
    seq_cols = [
        "B1-1","B1-2","B1-3",
        "B2-1","B2-2","B2-3",
        "B3-1","B3-2",
        "B4-1","B4-2",
        "B5-1","B5-2",
        "B6","B7","B8"
    ]
    print("B 검사 데이터 전처리 완료")
    out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out