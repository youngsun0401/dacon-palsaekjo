import numpy as np
import pandas as pd
from tqdm import tqdm
from .preprocess_util import convert_age, split_testdate, seq_mean, seq_std, seq_rate, masked_mean_from_csv_series, masked_mean_in_set_series

tqdm.pandas()


def preprocess_A(train_A: pd.DataFrame) -> pd.DataFrame:
    """
    A 검사용 원본 DataFrame을 받아 여러 요약 특징(feature)을 생성하고 시퀀스 컬럼을 제거하여 반환합니다.
    생성되는 주요 특징: Age_num, Year, Month 및 A1~A5 관련 반응률, 반응시간 평균/표준편차, 조건별 평균 등.
    무한대 값을 np.nan으로 치환합니다.

    Args:
        train_A: 원본 A 검사 DataFrame

    Returns:
        pd.DataFrame: 파생특징이 추가되고 시퀀스 컬럼이 제거된 DataFrame
    """
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
    '''
    여덟 가지 위치 중 랜덤으로 #가 나타난다.
    이후,
    여덟 가지 위치에 각각 표시가 나타난다.
      - 표시 중 하나는 화살표
      - 화살표가 #와 같은 위치일 확률 75%
      - 화살표는 좌 또는 우
      - 나머지 표시들은 화살표와 비슷하지만 아닌 함정들.

    화살표와 같은 방향의 버튼을 누르면 정답.

    - (1) 1:작 / 2:크
    - (2) 1-8 시계방향... #자의 위치인가?? 각 여덟 가지 경우는 네 번씩 나옴.(전체 검사는 32회=8*4)
    - (3) 1:왼 / 2:오 ... 정답
    - (4) 1-8 시계방향... 정답 화살표의 위치인가?? 이것도 네 번씩.
    - (5) 1: #가 정답위치, 정답  
          2: #가 정답위치, 오답  
          3: #가 틀린위치, 정답  
          4: #가 틀린위치, 오답  
    - (6) N / Y
    - (7) 반응시간
    '''
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

def preprocess_B(train_B: pd.DataFrame) -> pd.DataFrame:
    """
    B 검사용 원본 DataFrame을 받아 여러 요약 특징(feature)을 생성하고 시퀀스 컬럼을 제거하여 반환합니다.
    생성되는 주요 특징: Age_num, Year, Month 및 B1~B8 관련 정확도, 반응시간 평균/표준편차 등.
    무한대 값을 np.nan으로 치환합니다.

    Args:
        train_B: 원본 B 검사 DataFrame

    Returns:
        pd.DataFrame: 파생특징이 추가되고 시퀀스 컬럼이 제거된 DataFrame
    """
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