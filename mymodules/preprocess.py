import numpy as np
import pandas as pd
from tqdm import tqdm
from .preprocess_util import convert_age, split_testdate, seq_mean, seq_std, seq_rate, seq_skew, masked_mean_from_csv_series, masked_mean_in_set_series

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
                               masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 2)# 작은/큰 경우 평균 반응속도 차이
    feats["A3_rt_side_diff"] = masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 1) - \
                               masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 2)# 왼/오른 경우 평균 반응속도 차이

    print("Step 5: A4 feature 생성...")
    feats["A4_acc_rate"]   = seq_rate(df["A4-3"], "1")
    feats["A4_resp2_rate"] = seq_rate(df["A4-4"], "1")
    feats["A4_rt_mean"]    = seq_mean(df["A4-5"])
    feats["A4_rt_std"]     = seq_std(df["A4-5"])

    # ✅ 조건별 평균 RT (Congruent / Incongruent)
    feats["A4_rt_congruent"]   = masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1)
    feats["A4_rt_incongruent"] = masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 2)
    feats["A4_stroop_diff"]    = feats["A4_rt_incongruent"] - feats["A4_rt_congruent"]

    # ✅ 색상 조건별 반응 속도 차이
    feats["A4_rt_color_diff"] = (
        masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 1)
        - masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 2)
    )

    # ✅ 새로 추가되는 심화 feature
    # 1️⃣ 반응 시간의 변동계수 (RT 표준편차 / 평균)
    feats["A4_rt_cv"] = feats["A4_rt_std"] / (feats["A4_rt_mean"] + 1e-6)

    # 2️⃣ Stroop effect 비율 (차이 / 전체평균)
    feats["A4_stroop_ratio"] = feats["A4_stroop_diff"] / (feats["A4_rt_mean"] + 1e-6)

    # 3️⃣ 반응률 간의 상관 feature
    feats["A4_resp_acc_gap"] = feats["A4_acc_rate"] - feats["A4_resp2_rate"]

    # 4️⃣ 반응 속도 일관성: 표준편차 대비 평균 차이
    feats["A4_rt_stability"] = 1 / (feats["A4_rt_std"] + 1e-6)

    # 5️⃣ RT 극단값 비율 (RT의 길이 차이를 이용)
    feats["A4_rt_skewness"] = (
        (feats["A4_rt_incongruent"] - feats["A4_rt_mean"]) /
        (feats["A4_rt_std"] + 1e-6)
    )

# ---------------------------------------> A4 추가된 심화 feature 설명 <-------------------------------------

# 코드명                      역할                                     의미

# A4_rt_cv              반응 속도 기복         피험자가 매번 버튼을 누를 때마다 속도가 들쭉날쭉한 정도를 나타냅니다.
#                                            속도가 규칙적이지 않고 기복이 심하다면, 
#                                            그만큼 집중력이 불안정하다는 뜻이 됩니다.

# A4_stroop_ratio     헷갈림의 정도(상대적)    '헷갈려서 느려진 시간'을 평소 버튼 누르는 속도로 나눈 비율입니다. 
#                                            단순히 늦게 눌렀다가 아니라, 평소 실력 대비 헷갈림의 
#                                            충격이 얼마나 큰지를 측정합니다.

# A4_resp_acc_gap       신중함vs성급함         정확하게 맞춘 비율과 너무 늦게 눌러 실패한 비율의 차이입니다. 
#                                            이 간격이 크다는 건, 신중함과 성급함 사이에서 
#                                            반응의 밸런스를 잡지 못한다는 뜻입니다.
                                           
# A4_rt_stability         꾸준함'점수'         반응 속도의 기복을 뒤집은 점수입니다. 
#                                            이 점수가 높을수록 매우 꾸준하게 버튼을 잘 누른다는 뜻이며, 
#                                            일관된 집중력을 측정합니다.

# A4_rt_skewness    '유독 어려운 상황' 지표     가장 헷갈리는 상황(부일치 조건)의 RT가 보통의 속도에서 
#                                            얼마나 벗어났는지를 측정합니다. 
#                                            특정 상황에서만 인지 능력이 급격히 무너지는 패턴을 잡아냅니다.

# ---------------------------------------------------------------------------------------------------------

    print("Step 6: A5 feature 생성...")
    feats["A5_acc_rate"]   = seq_rate(df["A5-2"], "1")
    feats["A5_resp2_rate"] = seq_rate(df["A5-3"], "1")
    feats["A5_acc_nonchange"] = masked_mean_from_csv_series(df["A5-1"], df["A5-2"], 1)
    feats["A5_acc_change"]    = masked_mean_in_set_series(df["A5-1"], df["A5-2"], {2,3,4})
    print("Step 7: A6 feature 확장 생성...")
    

    df["A6-1"] = df["A6-1"].fillna("").astype(str)
    A6_df = df["A6-1"].str.split(",", expand=True)
    A6_arr = A6_df.replace("", np.nan).apply(pd.to_numeric, errors="coerce").to_numpy()

    mask_word_acc   = (A6_arr == 1)
    mask_figure_acc = (A6_arr == 2)
    mask_intrusion  = (A6_arr == 3)
    mask_nonrecall  = (A6_arr == 4)

    total_valid = np.sum(~np.isnan(A6_arr), axis=1).astype(float)
    total_valid[total_valid == 0] = np.nan

    # 기존 비율 Feature
    feats["A6_word_acc_rate"]   = np.nansum(mask_word_acc, axis=1) / total_valid
    feats["A6_figure_acc_rate"] = np.nansum(mask_figure_acc, axis=1) / total_valid
    feats["A6_intrusion_rate"]  = np.nansum(mask_intrusion, axis=1) / total_valid
    feats["A6_nonrecall_rate"]  = np.nansum(mask_nonrecall, axis=1) / total_valid

    # 파생 Feature
    feats["A6_total_acc_rate"] = feats["A6_word_acc_rate"] + feats["A6_figure_acc_rate"]
    feats["A6_mem_diff"]       = feats["A6_word_acc_rate"] - feats["A6_figure_acc_rate"]
    feats["A6_error_bias"]     = feats["A6_nonrecall_rate"] - feats["A6_intrusion_rate"]

    # ✅ 새로 추가되는 심화 feature
    # 1️⃣ 기억 성공률 대비 오류율 비율
    feats["A6_acc_vs_error"] = feats["A6_total_acc_rate"] / (
        feats["A6_intrusion_rate"] + feats["A6_nonrecall_rate"] + 1e-6
    )

    # 2️⃣ 회상 실패율 비율 (nonrecall / total)
    feats["A6_nonrecall_ratio"] = feats["A6_nonrecall_rate"] / (
        feats["A6_intrusion_rate"] + feats["A6_nonrecall_rate"] + 1e-6
    )

    # 3️⃣ 기억 과제 내 편향 지표 (intrusion - nonrecall)
    feats["A6_bias_index"] = feats["A6_intrusion_rate"] - feats["A6_nonrecall_rate"]

    # 4️⃣ 변동성 (A6 내 각 시도 간 일관성 지표)
    feats["A6_var"] = np.nanstd(A6_arr, axis=1)

    df["A6-1"] = df["A6-1"].astype(int)


# ---------------------------------------> A6 추가된 심화 feature 설명 <-------------------------------------

# 코드명                      역할                                     의미

# A6_acc_vs_error         기억의 정확도        '성공적으로 기억한 횟수'를 '실수한 횟수'로 나눈 값입니다.
#                                            이 숫자가 높을수록 실수는 적게 하면서 잘 맞추는, 
#                                            매우 뛰어난 기억력을 가졌다는 뜻입니다.

# A6_nonrecall_ratio      주요실수유형         전체 실수 중에서 '아예 기억을 못 해서 빈칸으로 둔 실수'가 차지하는 비율입니다. 
#                                            이 비율이 높으면, 모르는 것을 대충 말하기보다는 
#                                            조심스럽게 침묵하는 경향을 보입니다.

# A6_bias_index          실수하는 태도         '틀린것을 채워 넣는 실수'와 '기억 못해서 비워둔 실수' 차이입니다.
#                                            (+)숫자가 크면: 성급하게 대충찍는 스타일입니다.
#                                            (-)숫자가 적으면: 신중하게 확실한 것만 말하는 스타일입니다.
                                           
# A6_var                 집중력의 변화         기억 과제를 하는 동안 피험자의 반응이 얼마나 바뀌었는지를 보여주는 값입니다. 
#                                            이 값이 높으면, 어떤 문제에서는 잘하다가 어떤 문제에서는 
#                                            크게 실수하는 등 집중력이나 실력이 일정하지 않고 기복이 심하다는 뜻입니다.

# ---------------------------------------------------------------------------------------------------------

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

    '''
    B1, B2 검사의 설명이 완전히 같다.

    (1) 1번 과제의 1: 정답 / 2: 오답
    (2) 2번 과제의 응답시간(초)
    (3) 2번 과제의
      - 1: 문제의 색 바뀜, 정답
      - 2: 문제의 색 바뀜, 오답
      - 3: 문제의 색 안 바뀜, 정답
      - 4: 문제의 색 안 바뀜, 오답
    '''
    print("Step 2: B1 feature 생성...")
    feats["B1_acc_task1"] = seq_rate(df["B1-1"], "1") # 과제1 정확도
    feats["B1_rt_mean"]   = seq_mean(df["B1-2"])      # 과제2 반응속도 평균
    feats["B1_rt_std"]    = seq_std(df["B1-2"])       # 과제2 반응속도 표준편차
    feats["B1_acc_task2"] = seq_rate(df["B1-3"], "1") # 과제2 정확도
    feats["B1_rt_skew"]   = seq_skew(df["B1-2"])      # 반응속도의 왜도

    print("Step 3: B2 feature 생성...")
    feats["B2_acc_task1"] = seq_rate(df["B2-1"], "1") # 과제1 정확도
    feats["B2_rt_mean"]   = seq_mean(df["B2-2"])      # 과제2 반응속도 평균
    feats["B2_rt_std"]    = seq_std(df["B2-2"])       # 과제2 반응속도 표준편차
    feats["B2_acc_task2"] = seq_rate(df["B2-3"], "1") # 과제2 정확도
    feats["B2_rt_skew"]   = seq_skew(df["B2-2"])      # 반응속도의 왜도

    print("Step 4: B3 feature 생성...")
    feats["B3_acc_rate"] = seq_rate(df["B3-1"], "1")
    feats["B3_rt_mean"]  = seq_mean(df["B3-2"])
    feats["B3_rt_std"]   = seq_std(df["B3-2"])

    print("Step 5: B4 feature 생성...")
    feats["B4_acc_rate"] = seq_rate(df["B4-1"], "1")
    feats["B4_rt_mean"]  = seq_mean(df["B4-2"])
    feats["B4_rt_std"]   = seq_std(df["B4-2"])

    print("Step 6: B5 feature 확장 생성...")

    # B5-1: 응답 (1: 정답, 2: 오답), B5-2: 반응시간
    # 응답 1이 정답이라고 가정합니다.
    feats["B5_acc_rate"]    = seq_rate(df["B5-1"], "1")    # 정답률 (Accuracy Rate)
    feats["B5_incorrect_rate"] = seq_rate(df["B5-1"], "2")    # 오답률 (Incorrect Rate)
    feats["B5_rt_mean"]     = seq_mean(df["B5-2"])     # 전체 평균 반응 시간
    feats["B5_rt_std"]      = seq_std(df["B5-2"])      # 반응 시간의 표준 편차 (기복)

    # ✅ 심화 Feature (A4 패턴 활용)

    # 1️⃣ 반응 시간의 변동계수 (RT 표준편차 / 평균): 집중력의 불안정성 측정
    feats["B5_rt_cv"] = feats["B5_rt_std"] / (feats["B5_rt_mean"] + 1e-6)

    # 2️⃣ 정확도 대비 속도 비율: 정확도를 얼마나 포기하고 속도를 냈는지 측정 (Trade-off)
    # RT가 빠를수록 B5_speed_ratio는 작아짐 -> (1 / RT) * ACC
    feats["B5_speed_ratio"] = feats["B5_acc_rate"] / (feats["B5_rt_mean"] + 1e-6)

    # 3️⃣ 반응 속도 일관성: 표준편차의 역수 (A4_rt_stability와 동일)
    feats["B5_rt_stability"] = 1 / (feats["B5_rt_std"] + 1e-6)

    # 4️⃣ 정확하게 맞춘 문제의 평균 RT
    feats["B5_rt_correct_mean"] = masked_mean_from_csv_series(df["B5-1"], df["B5-2"], 1)

    # 5️⃣ 정답률과 오답률의 차이 (집중도/명확성)
    feats["B5_acc_error_gap"] = feats["B5_acc_rate"] - feats["B5_incorrect_rate"]

# ---------------------------------------> B5 추가된 심화 feature 설명 <-------------------------------------

# 코드명                      역할                                     의미

# B5_acc_rate           공간 판단 정확도             20번의 시도 중 올바른 길을 맞춘 비율입니다. 
#                                                  높을수록 공간 판단 능력이 뛰어납니다.

# B5_rt_mean             평균 반응 속도              문제를 푸는 데 걸린 평균 시간입니다. 
#                                                  짧을수록 빠른 인지 속도를 의미합니다.

# B5_rt_cv               판단 속도 기복              문제를 풀 때마다 속도가 얼마나 들쭉날쭉했는지를 측정합니다. 
#                                                  이 값이 높으면 집중력이 불안정하다는 뜻입니다.
                                           
# B5_speed_ratio         수행 효율성                정확도를 평균 반응 시간으로 나눈 값입니다. 
#                                                 이 값이 높을수록 빠르면서도 정확하게 
#                                                 문제를 푸는 인지 효율성이 높다는 의미입니다.

# B5_rt_correct_mean    정답 시 평균속도            정답을 맞췄을 때만의 평균 반응 시간입니다. 
#                                                 오답을 포함한 전체 평균보다 순수한 공간 판단 속도를 
#                                                 더 잘 반영합니다.

# B5_acc_error_gap        판단 명확성              정답률과 오답률의 차이입니다. 이 차이가 클수록 명확하게 
#                                                정답을 식별하고 있다는 것을 의미합니다.

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