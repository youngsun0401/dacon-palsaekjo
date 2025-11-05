import os, joblib
import numpy as np
import pandas as pd

from mymodules.preprocess import preprocess_A, preprocess_B
from mymodules.feature_engineering import add_features_A, add_features_B

from mymodules.paths import Paths
PATH = Paths()
print(PATH)

# ---- 모델 불러오기 ----
print("Load models...")
model_A = joblib.load(PATH.model_A)
model_B = joblib.load(PATH.model_B)

attrs_A = getattr(model_A, "feature_name_", [])
attrs_B = getattr(model_B, "feature_name_", [])
print(" OK.")

# ---- 테스트 데이터 불러오기 ----
print("Load test data...")
meta = pd.read_csv(PATH.test_meta)
Araw = pd.read_csv(PATH.test_A)
Braw = pd.read_csv(PATH.test_B)
print(f" meta={len(meta)}, Araw={len(Araw)}, Braw={len(Braw)}")

# ---- 매핑 ----
A_df = meta.loc[meta["Test"] == "A", ["Test_id", "Test"]].merge(Araw, on="Test_id", how="left")
B_df = meta.loc[meta["Test"] == "B", ["Test_id", "Test"]].merge(Braw, on="Test_id", how="left")
print(f" mapped: A={len(A_df)}, B={len(B_df)}")

# ---- 전처리 → 파생 (학습과 동일) ----
A_feat = add_features_A(preprocess_A(A_df)) if len(A_df) else pd.DataFrame()
B_feat = add_features_B(preprocess_B(B_df)) if len(B_df) else pd.DataFrame()

# =======================
# 정렬/보정 (모델이 학습 때 본 피처 순서로)
# =======================
COLS_TO_DROP = ["Test_id","Test","PrimaryKey","Age","TestDate"]

def align_to_model(X_df, attrs):
    feat_names = list(attrs)
    if not feat_names:
        # fallback: 그냥 숫자형만
        X = X_df.select_dtypes(include=[np.number]).copy()
        return X.fillna(0.0)
    X = X_df.drop(columns=[c for c in COLS_TO_DROP if c in X_df.columns], errors="ignore").copy()
    # 누락 피처 0으로 채움
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0.0
    # 초과 피처 드롭 + 순서 일치
    X = X[feat_names]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    if len(X_df):
        return X
    else:
        pd.DataFrame(columns=attrs)

# ---- 피처 정렬/보정 ----
XA = align_to_model(A_feat, attrs_A)
XB = align_to_model(B_feat, attrs_B)
print(f" aligned: XA={XA.shape}, XB={XB.shape}")

# ---- 예측 ----
print("Inference Model...")
predA = model_A.predict_proba(XA)[:,1] if len(XA) else np.array([])
predB = model_B.predict_proba(XB)[:,1] if len(XB) else np.array([])

subA  = pd.DataFrame({"Test_id": A_df["Test_id"].values, "Label": predA})
subB  = pd.DataFrame({"Test_id": B_df["Test_id"].values, "Label": predB})
probs = pd.concat([subA, subB], axis=0, ignore_index=True).sort_values('Test_id')

print(len(probs))
# display(probs)

probs.to_csv(PATH.submission, index=False)
print(f"결과 저장: " + PATH.submission)