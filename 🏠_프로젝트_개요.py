import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Streamlit 페이지 설정
st.set_page_config(
    page_title="당뇨병 위험군 예측 및 자가진단",
    page_icon="🩺",
    layout="wide"
)

st.markdown("# 🩺 당뇨병 위험군 예측 및 자가진단 프로젝트")
st.info("""
본 서비스는 국민건강보험공단의 건강검진 빅데이터를 기반으로 **당뇨병 위험군 여부를 예측하는 머신러닝 웹 서비스**입니다.  
**왼쪽 측면 메뉴의 Prediction 페이지**에서 본인의 건강검진 수치를 입력하시면, 머신러닝 모델이 분석한 **당뇨병 위험도 및 맞춤형 예측 결과**를 확인하실 수 있습니다.
""")


col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 프로젝트 개요
    
    | 구분 | 세부 내용 |
    |------|-------------|
    | **분석 주제** | 건강검진 데이터 기반 당뇨병 위험군 예측 |
    | **수행 과제** | 이진 분류 (저위험군 / 고위험군 예측) |
    | **수집 데이터** | [국민건강보험공단 건강검진정보](https://data.edmgr.kr/dataView.do?id=www-data-go-kr-data-filedata-15007122) |
    | **개발 환경** | Streamlit (파이썬 기반 웹 프레임워크) |
    """)

with col2:
    st.markdown("""
    #### 주요 개발 목표

    - 건강검진 데이터 탐색적 분석(EDA) 및 전처리
    - 최적의 머신러닝 모델 구축 및 성능 비교·평가
    - 실시간 사용자 입력 기반의 당뇨병 자가진단 인터페이스 구현
    - 예측 결과에 따른 사용자 맞춤형 건강 가이드 제공
    """)

st.markdown("")
st.markdown("#### 프로젝트 수행 절차 (Workflow)")

# 6개 단계를 가로 컬럼으로 배치
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown("##### 1. 데이터 수집")
    st.caption("건강보험공단 검진 데이터")

with c2:
    st.markdown("##### 2. 데이터 전처리")
    st.caption("결측치 및 이상치 정제")

with c3:
    st.markdown("##### 3. EDA")
    st.caption("주요 지표 분포 탐색")

with c4:
    st.markdown("##### 4. 모델 비교")
    st.caption("4개 머신러닝 알고리즘")

with c5:
    st.markdown("##### 5. 최종 모델")
    st.caption("XGBoost 파라미터 튜닝")

with c6:
    st.markdown("##### 6. 서비스 구현")
    st.caption("Streamlit 웹 UI")


st.markdown("---")
st.markdown("## 탐색적 데이터 분석 (EDA)")
st.markdown("""
건강검진 데이터를 기반으로 **당뇨병 고위험군**에 영향을 미치는 주요 건강 지표와 생활습관 요인을 분석했습니다.
""")
st.badge("💡 공복혈당 100mg/dL 이상을 기준으로 분석", color="blue")

# 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터 불러오기
df = pd.read_csv('diabetes_preprocessed.csv')

col3, col4, col4_2 = st.columns([3,4,2])

with col3:
    # 타겟 비율

    target_counts = (
        df['Target']
        .value_counts()
        .sort_index()
        .rename(index={0: '저위험군', 1: '고위험군'})
        )

    target_df = target_counts.reset_index()
    target_df.columns = ["분류", "인원 수"]
    total = target_df["인원 수"].sum()

    fig = px.pie(
        target_df,
        names="분류",
        values="인원 수",
        hole=0.7,   # 도넛
        title="당뇨병 저위험군 및 고위험군 분포",
        color="분류",
        color_discrete_map={
            "저위험군": "#2E8B57",   # 초록
            "고위험군": "#ff4b4b"    # 빨강
        }
    )

    fig.update_traces(
        texttemplate="%{label}<br>%{value:,}명<br>(%{percent})",
        textfont_size=14
    )

    fig.update_layout(
        showlegend=False, 
        annotations=[
            dict(
                text=f"<b>{total:,}명</b><br>전체",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
        ]
    )



    st.plotly_chart(fig, use_container_width=True)

with col4:
    # 연령별 위험군 비율

    age_risk = (
        df.groupby('연령대코드(5세단위)', as_index=False)
        .agg(
            위험군비율=('Target', 'mean'),
            전체인원=('Target', 'size')
        )
    )

    age_risk['고위험군비율_퍼센트'] = age_risk['위험군비율'] * 100

    fig = px.bar(
        age_risk,
        x="연령대코드(5세단위)",
        y="고위험군비율_퍼센트",
        text="고위험군비율_퍼센트",
        title="연령대별 당뇨병 고위험군 비율"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_color="#2E8B57"
    )

    fig.update_layout(
        xaxis_title="연령대 코드(5세 단위)",
        yaxis_title="고위험군 비율(%)",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

with col4_2:
    # 흡연 상태별 위험군 비율

    smoking_risk = (
        df.groupby("흡연상태", as_index=False)
        .agg(
            위험군비율=("Target", "mean"),
            전체인원=("Target", "size")
        )
    )

    smoking_risk["흡연구분"] = smoking_risk["흡연상태"].map({
        1: "비흡연",
        2: "과거 흡연",
        3: "현재 흡연"
    })

    smoking_risk["고위험군비율_퍼센트"] = smoking_risk["위험군비율"] * 100

    smoking_risk["흡연구분"] = pd.Categorical(
        smoking_risk["흡연구분"],
        categories=["비흡연", "과거 흡연", "현재 흡연"],
        ordered=True
    )

    smoking_risk = smoking_risk.sort_values("흡연구분")

    fig = px.bar(
        smoking_risk,
        x="흡연구분",
        y="고위험군비율_퍼센트",
        text="고위험군비율_퍼센트",
        title="흡연상태에 따른 당뇨병 고위험군 비율",
        color="흡연구분",
        color_discrete_map={
            "비흡연": "#2E8B57",      # 초록
            "과거 흡연": "#FFA500",    # 주황
            "현재 흡연": "#ff4b4b"     # 빨강
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="흡연상태",
        yaxis_title="고위험군 비율(%)",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

with st.expander("📖 그래프 해석"):
    st.markdown("""
- **저위험군과 고위험군의 비율은 약 57:43**으로, 심한 클래스 불균형 없이 비교적 균형 있는 분포를 보였습니다.
- **연령이 증가할수록 당뇨병 고위험군 비율이 전반적으로 증가**하는 경향을 확인할 수 있습니다.
- **과거 흡연자와 현재 흡연자**에서 비흡연자보다 높은 고위험군 비율이 나타났습니다.
""")

plot_df = df.copy()
plot_df["그룹"] = plot_df["Target"].map({
    0: "저위험군",
    1: "고위험군"
})

col5, col6, col7 = st.columns(3)

with col5:
    fig = px.box(
        plot_df,
        x="그룹",
        y="BMI",
        color="그룹",
        title="저위험군과 고위험군의 BMI 분포 비교",
        color_discrete_map={
            "저위험군": "#2E8B57",
            "고위험군": "#ff4b4b"
        },
        points=False
    )

    fig.update_layout(
        height=500,   # 세로 길이
        xaxis_title="",
        yaxis_title="BMI",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

with col6:
    fig = px.box(
        plot_df,
        x="그룹",
        y="수축기혈압",
        color="그룹",
        title="저위험군과 고위험군의 수축기혈압 분포 비교",
        color_discrete_map={
            "저위험군": "#2E8B57",
            "고위험군": "#ff4b4b"
        },
        points=False
    )

    fig.update_layout(
        height=500,   # 세로 길이
        xaxis_title="",
        yaxis_title="수축기혈압",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

with col7:
    fig = px.box(
        plot_df,
        x="그룹",
        y="이완기혈압",
        color="그룹",
        title="저위험군과 고위험군의 이완기혈압 분포 비교",
        color_discrete_map={
            "저위험군": "#2E8B57",
            "고위험군": "#ff4b4b"
        },
        points=False
    )

    fig.update_layout(
        height=500,   # 세로 길이
        xaxis_title="",
        yaxis_title="이완기혈압",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

with st.expander("📖 그래프 해석"):
    st.markdown("""
- 고위험군은 저위험군보다 **BMI가 전반적으로 높은 분포**를 보였습니다.
- **수축기혈압과 이완기혈압 역시 고위험군에서 더 높은 경향**을 확인할 수 있습니다.
- BMI와 혈압은 당뇨병 위험을 예측하는 주요 건강 지표로 활용될 수 있음을 확인했습니다.
""")

col8, col9 = st.columns(2)

with col8:
    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    fig.update_layout(
        title="주요 건강 지표 간 상관관계 분석 (히트맵)",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

with col9:

    target_corr = (
        df.corr(numeric_only=True)["Target"]
        .drop("Target")
        .sort_values(ascending=False)
    )

    corr_df = target_corr.reset_index()
    corr_df.columns = ["변수", "상관계수"]

    fig = px.bar(
        corr_df,
        x="상관계수",
        y="변수",
        orientation="h",
        text="상관계수",
        color="상관계수",
        color_continuous_scale="RdBu_r",
        title="건강 검진 항목별 당뇨 위험도(타겟) 관련성"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="상관계수",
        yaxis_title="",
        coloraxis_showscale=False,
        height=550
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_color="black"
    )

    st.plotly_chart(fig, use_container_width=True)

with st.expander("📖 그래프 해석"):
    st.markdown("""
- 대부분의 변수는 **강한 상관관계를 보이지 않아**, 다양한 정보를 활용한 모델 학습이 가능함을 확인했습니다.
- Target과의 상관관계에서는 **허리둘레, 연령, BMI, 수축기혈압, 중성지방** 등이 상대적으로 높은 관련성을 보였습니다.
- 단일 변수보다 **여러 건강 지표를 종합적으로 활용하는 머신러닝 접근이 적합**함을 확인했습니다.
""")

st.markdown("----")
st.markdown("## 머신러닝 모델 비교 및 성능 평가")


st.markdown("### 📊 4가지 주요 모델 성능 비교")

my_blues = ["#0047AB", "#1F75FE", "#73C2FB", "#BDE0FE"]

col10, col11, col12=st.columns(3)

with col10:
    df_score = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"],
        "정확도": [0.667518, 0.661426, 0.651434, 0.644420]
    })

    fig = px.bar(
        df_score, 
        x="Model",
        y="정확도",
        text="정확도",
        color="Model",
        color_discrete_sequence=my_blues
    )

    fig.update_traces(texttemplate="%{text:.3f}")

    fig.update_layout(
        yaxis_range=[0.6, 0.7],
        height=450,
        legend_title="정확도",
        showlegend=False,
        title="정확도 (Accuracy)"
    )

    st.plotly_chart(fig, use_container_width=True)

with col11:
    df_score = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"],
        "정밀도": [0.663944, 0.657157, 0.646557, 0.639324]
    })

    fig = px.bar(
        df_score,
        x="Model",
        y="정밀도",
        text="정밀도",
        color="Model",
        color_discrete_sequence=my_blues
    )

    fig.update_traces(texttemplate="%{text:.3f}")

    fig.update_layout(
        yaxis_range=[0.6, 0.7],
        height=450,
        legend_title="정밀도",
        showlegend=False,
        title="정밀도 (Weighted Precision)"
    )

    st.plotly_chart(fig, use_container_width=True)

with col12:
    df_score = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"],
        "F1-스코어": [0.663826, 0.655587, 0.644533, 0.638237]
    })

    fig = px.bar(
        df_score,
        x="Model",
        y="F1-스코어",
        text="F1-스코어",
        color="Model",
        color_discrete_sequence=my_blues
    )

    fig.update_traces(texttemplate="%{text:.3f}")

    fig.update_layout(
        yaxis_range=[0.6, 0.7],
        height=450,
        legend_title="F1-스코어",
        showlegend=False,
        title="F1-스코어(Weighted F1-score)"
    )

    st.plotly_chart(fig, use_container_width=True)

with st.expander("📖 성능 비교 그래프 해석"):
    st.markdown("""
- **정확도(Accuracy)**: XGBoost가 가장 높은 예측 정확도를 보였으며, Random Forest가 그 뒤를 이었습니다.
- **정밀도(Precision)**: XGBoost가 가장 높은 정밀도를 기록하여 오분류(False Positive)를 가장 효과적으로 줄였습니다.
- **F1-score**: 정밀도와 재현율을 종합적으로 평가한 결과에서도 XGBoost가 가장 우수한 성능을 보였습니다.
""")

st.markdown("")

st.markdown("### 🏆 최종 예측 모델 선정: XGBoost")

a, b, c, d = st.columns([2,2,2,3])
a.metric("정확도 (Accuracy)", "0.667", border=True)
b.metric("재현율 (Weighted Recall)", "0.663", border=True)
c.metric("F1-스코어 (Weighted F1-score)", "0.663", border=True)
st.caption("Gradient Boosting 기반의 앙상블 모델인 XGBoost는 변수 간의 복잡한 상호작용을 효과적으로 학습하여 모든 평가 지표(정확도, 정밀도, F1-스코어)에서 가장 뛰어난 성능을 입증했습니다.")

st.markdown("")

st.markdown("### ⚙️ 하이퍼파라미터 최적화")

st.markdown("#### 최적 파라미터 조합")
left, right=st.columns([1,2])

with left:
    bestparameter_df=pd.DataFrame({
        'Parameter':["`'max_depth'`", "`'learning_rate'`", "`'n_estimators'`", "`'subsample'`", "`'colsample_bytree'`"],
        'Value':['5', '0.1', '200', '1.0', '0.8']
    })
    st.table(bestparameter_df)
st.caption("💡 GridSearchCV 및 5-Fold Cross Validation을 통해 총 5가지 핵심 파라미터를 최적화하여 모델 성능을 추가로 향상시켰습니다.")
st.markdown("")

st.markdown("#### 최적화 후 최종 모델 성능")

e, f, g, h = st.columns([2,2,2,3])
e.metric("정확도 (Accuracy)", "0.6694", border=True)
f.metric("F1-스코어 (Weighted F1-score)", "0.6660", border=True)
g.metric("정밀도 (Weighted Precision)", "0.6660", border=True)


col13, col14, col15 = st.columns([6,10,2])
    
with col13:
    cm_data = [[28715, 9266], 
            [12603, 15567]]

    x_labels = ["예측: 저위험군", "예측: 고위험군"]
    y_labels = ["실제: 저위험군", "실제: 고위험군"]

    fig = px.imshow(
        cm_data,
        x=x_labels,
        y=y_labels,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto"
    )

    fig.update_layout(
        title="혼동 행렬 (Confusion Matrix)",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

with col14:
    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.info("""
    📊 **오분류 경향성 분석**

    - 저위험군(0)을 정확하게 분류한 비율이 상대적으로 높으며, 고위험군(1) 역시 안정적으로 구분해 내고 있습니다.

    - 모델이 예측할 때 발생시키는 오분류의 위치와 경향성을 한눈에 확인할 수 있습니다.
    """)


st.markdown("**클래스별 평가지표 (Classification Report)**")
col16, col17, col18 = st.columns([6,10,2])
with col16:
    df_report = pd.DataFrame(
        [
            [0.69, 0.76, 0.72],
            [0.63, 0.55, 0.59]
        ],
        columns=["정밀도", "재현율", "F1-스코어"],
        index=["저위험군 (0)", "고위험군 (1)"]
    )

    st.dataframe(
        df_report.style.format("{:.2f}").background_gradient(cmap="Blues", vmin=0.5, vmax=1.0),
        use_container_width=True
    )
with col17:
    st.info("""
    📋 **클래스별 세부 성능**

    - 저위험군(0): 정밀도 0.69 / 재현율 0.76으로 안정적인 성능을 보입니다.

    - 고위험군(1): 재현율(0.55)이 상대적으로 낮아 일부 고위험군을 저위험군으로 예측하는 경우가 존재하므로, 향후 데이터 보강 및 모델 개선의 주요 포인트로 활용할 수 있습니다.
    """)