import pandas as pd
import numpy as np
import joblib
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
 
 
# XGBoost 버전 확인
print("XGBoost version:", xgboost.__version__)
 
 
# 데이터 불러오기
df = pd.read_csv("diabetes_preprocessed.csv")
 
print("데이터 크기:", df.shape)
print(df.head())
 
 
# Target 설정
TARGET = "Target"
 
X = df.drop(TARGET, axis=1)
y = df[TARGET]
 
 
# 결측치 처리 (X만 평균으로 대체)
X = X.fillna(X.mean(numeric_only=True))
 
 
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
 
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
 
 
# 팀원이 GridSearchCV로 이미 찾은 최적 하이퍼파라미터를 그대로 사용
best_params = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "max_depth": 5,
    "n_estimators": 200,
    "subsample": 1.0
}

print("\n===============================")
print("최적 하이퍼파라미터")
print("===============================")
print(best_params)
 
best_model = XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric="mlogloss"
)
 
print("\n모델 학습 시작...")
best_model.fit(X_train, y_train)
print("모델 학습 완료!")
 
 
# 테스트 데이터로 최종 성능 평가
best_pred = best_model.predict(X_test)
 
best_acc = accuracy_score(y_test, best_pred)
 
best_pre = precision_score(
    y_test,
    best_pred,
    average="weighted"
)
 
best_rec = recall_score(
    y_test,
    best_pred,
    average="weighted"
)
 
best_f1 = f1_score(
    y_test,
    best_pred,
    average="weighted"
)
 
 
print("\n===============================")
print("튜닝 후 XGBoost 최종 성능")
print("===============================")
 
print(f"Accuracy : {best_acc:.4f}")
print(f"Precision: {best_pre:.4f}")
print(f"Recall   : {best_rec:.4f}")
print(f"F1-score : {best_f1:.4f}")
 
 
print("\nClassification Report")
print(classification_report(y_test, best_pred))
 
 
print("\nConfusion Matrix")
print(confusion_matrix(y_test, best_pred))
 
 
joblib.dump(best_model, "model.pkl")