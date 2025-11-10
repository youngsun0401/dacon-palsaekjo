import numpy as np
import pandas as pd

def convert_age(val):
    """
    주어진 연령 문자열을 숫자로 변환합니다.
    입력 예: "30a" -> 30, "30b" 등(끝 문자가 "a"가 아니면) -> base+5.
    유효하지 않거나 결측이면 np.nan을 반환합니다.

    Args:
        val: 연령을 나타내는 값(문자열 등)

    Returns:
        float or int: 변환된 연령 또는 np.nan
    """
    if pd.isna(val): return np.nan
    try:
        base = int(str(val)[:-1])
        return base if str(val)[-1] == "a" else base + 5
    except:
        return np.nan

def split_testdate(val):
    """
    TestDate 값을 연도와 월로 분리합니다.
    정수형 YYYYMM 형태를 예상하며, 실패 시 (np.nan, np.nan)을 반환합니다.

    Args:
        val: 연도와 월이 결합된 값(예: 202301)

    Returns:
        tuple: (year, month) 또는 (np.nan, np.nan)
    """
    try:
        v = int(val)
        return v // 100, v % 100
    except:
        return np.nan, np.nan

def seq_mean(series):
    """
    콤마로 구분된 수열 문자열 시리즈에 대해 각 행의 평균을 계산합니다.
    빈 문자열 또는 결측은 np.nan으로 처리합니다. tqdm 진행바를 사용합니다.

    Args:
        series: 문자열 시리즈 (예: "1.0,2.0,3.0")

    Returns:
        pd.Series: 각 행의 평균값 (인덱스 보존)
    """
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )

def seq_std(series):
    """
    콤마로 구분된 수열 문자열 시리즈에 대해 각 행의 표준편차(std)를 계산합니다.
    빈 문자열 또는 결측은 np.nan으로 처리합니다. tqdm 진행바를 사용합니다.

    Args:
        series: 문자열 시리즈 (예: "1.0,2.0,3.0")

    Returns:
        pd.Series: 각 행의 표준편차 (인덱스 보존)
    """
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )

def seq_rate(series, target="1"):
    """
    콤마로 구분된 응답 시퀀스에서 특정 값(target)의 비율을 계산합니다.
    빈 문자열 또는 결측이면 np.nan을 반환합니다.

    Args:
        series: 문자열 시리즈 (예: "1,0,1,1")
        target: 비율을 계산할 값 (문자열, 기본 "1")

    Returns:
        pd.Series: 각 행에 대한 target 비율 (인덱스 보존)
    """
    return series.fillna("").progress_apply(
        lambda x: str(x).split(",").count(target) / len(x.split(",")) if x else np.nan
    )

def masked_mean_from_csv_series(cond_series, val_series, mask_val):
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

def masked_mean_in_set_series(cond_series, val_series, mask_set):
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