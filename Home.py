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
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

st.markdown("# 🩺 Diabetes Risk Assessment")
st.info("""
건강검진 데이터를 기반으로 당뇨병 위험군 여부를 예측하는 머신러닝 웹서비스입니다.  
사용자가 건강검진 정보를 입력하면 당뇨병 위험군 여부와 예측 결과를 확인할 수 있습니다.
""")


col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Project Information
    
    | Item | Description |
    |------|-------------|
    | **Problem** | Diabetes Risk Group Prediction |
    | **Task** | Binary Classification |
    | **Dataset** | [국민건강보험공단 건강검진정보](https://data.edmgr.kr/dataView.do?id=www-data-go-kr-data-filedata-15007122) |
    | **Framework** | Streamlit |
    """)

with col2:
    st.markdown("""
    #### Project Goals

    - 데이터 분석 및 전처리
    - 머신러닝 기반 당뇨병 위험군 예측
    - 모델 비교 및 성능 평가
    - Streamlit 기반 자가진단 웹서비스 구현
    """)


st.markdown("----")
st.markdown("## Exploratory Data Analysis (EDA)")


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
        .rename(index={0: '정상군', 1: '위험군'})
        )

    target_df = target_counts.reset_index()
    target_df.columns = ["분류", "인원 수"]
    total = target_df["인원 수"].sum()

    fig = px.pie(
        target_df,
        names="분류",
        values="인원 수",
        hole=0.7,   # 도넛
        title="당뇨병 정상군 및 위험군 분포",
        color="분류",
        color_discrete_map={
            "정상군": "#2E8B57",   # 초록
            "위험군": "#ff4b4b"    # 빨강
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

    age_risk['위험군비율_퍼센트'] = age_risk['위험군비율'] * 100

    fig = px.bar(
        age_risk,
        x="연령대코드(5세단위)",
        y="위험군비율_퍼센트",
        text="위험군비율_퍼센트",
        title="연령대별 당뇨병 위험군 비율"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_color="#2E8B57"
    )

    fig.update_layout(
        xaxis_title="연령대 코드(5세 단위)",
        yaxis_title="위험군 비율(%)",
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

    smoking_risk["위험군비율_퍼센트"] = smoking_risk["위험군비율"] * 100

    smoking_risk["흡연구분"] = pd.Categorical(
        smoking_risk["흡연구분"],
        categories=["비흡연", "과거 흡연", "현재 흡연"],
        ordered=True
    )

    smoking_risk = smoking_risk.sort_values("흡연구분")

    fig = px.bar(
        smoking_risk,
        x="흡연구분",
        y="위험군비율_퍼센트",
        text="위험군비율_퍼센트",
        title="흡연상태에 따른 당뇨병 위험군 비율",
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
        yaxis_title="위험군 비율(%)",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

plot_df = df.copy()
plot_df["그룹"] = plot_df["Target"].map({
    0: "정상군",
    1: "위험군"
})

col5, col6, col7 = st.columns(3)

with col5:
    fig = px.box(
        plot_df,
        x="그룹",
        y="BMI",
        color="그룹",
        title="정상군과 위험군의 BMI 분포 비교",
        color_discrete_map={
            "정상군": "#2E8B57",
            "위험군": "#ff4b4b"
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
        title="정상군과 위험군의 수축기혈압 분포 비교",
        color_discrete_map={
            "정상군": "#2E8B57",
            "위험군": "#ff4b4b"
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
        title="정상군과 위험군의 이완기혈압 분포 비교",
        color_discrete_map={
            "정상군": "#2E8B57",
            "위험군": "#ff4b4b"
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
        title="Feature Correlation Heatmap"
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
        title="각 변수와 Target의 상관관계"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="상관계수",
        yaxis_title="",
        coloraxis_showscale=False
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_color="black"
    )

    st.plotly_chart(fig, use_container_width=True)


st.markdown("----")
st.markdown("## Model Comparison & Performance")


st.markdown("### Model Comparison")

my_blues = ["#0047AB", "#1F75FE", "#73C2FB", "#BDE0FE"]

col10, col11, col12=st.columns(3)

with col10:
    df_score = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"],
        "Accuracy (Weighted Recall)": [0.667518, 0.661426, 0.651434, 0.644420]
    })

    fig = px.bar(
        df_score, 
        x="Model",
        y="Accuracy (Weighted Recall)",
        text="Accuracy (Weighted Recall)",
        color="Model",
        color_discrete_sequence=my_blues
    )

    fig.update_traces(texttemplate="%{text:.3f}")

    fig.update_layout(
        yaxis_range=[0.6, 0.7],
        height=450,
        legend_title="Accuracy (Weighted Recall)",
        showlegend=False,
        title="Accuracy (Weighted Recall)"
    )

    st.plotly_chart(fig, use_container_width=True)

with col11:
    df_score = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"],
        "Weighted Precision": [0.663944, 0.657157, 0.646557, 0.639324]
    })

    fig = px.bar(
        df_score,
        x="Model",
        y="Weighted Precision",
        text="Weighted Precision",
        color="Model",
        color_discrete_sequence=my_blues
    )

    fig.update_traces(texttemplate="%{text:.3f}")

    fig.update_layout(
        yaxis_range=[0.6, 0.7],
        height=450,
        legend_title="Weighted Precision",
        showlegend=False,
        title="Weighted Precision"
    )

    st.plotly_chart(fig, use_container_width=True)

with col12:
    df_score = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"],
        "F1-score": [0.663826, 0.655587, 0.644533, 0.638237]
    })

    fig = px.bar(
        df_score,
        x="Model",
        y="F1-score",
        text="F1-score",
        color="Model",
        color_discrete_sequence=my_blues
    )

    fig.update_traces(texttemplate="%{text:.3f}")

    fig.update_layout(
        yaxis_range=[0.6, 0.7],
        height=450,
        legend_title="F1-score",
        showlegend=False,
        title="F1-score"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Selected Model: XGBoost")
st.markdown("XGBoost가 Accuracy, Weighted Precision, Weighted F1-score에서 가장 우수한 성능(Performance)을 보여 최종 모델로 선정하였습니다.")

a, b, c, d = st.columns([2,2,2,3])
a.metric("Accuracy (Weighted Recall)", "0.667", border=True)
b.metric("Weighted Precision", "0.663", border=True)
c.metric("Weighted F1-score", "0.663", border=True)
st.markdown("")

st.markdown("### Hyperparameter Optimization")

st.markdown("#### Best Parameter")

left, right=st.columns([1,2])

with left:
    bestparameter_df=pd.DataFrame({
        'Parameter':["`'max_depth'`", "`'learning_rate'`", "`'n_estimators'`", "`'subsample'`", "`'colsample_bytree'`"],
        'Value':['5', '0.1', '200', '1.0', '0.8']
    })
    st.table(bestparameter_df)
st.markdown("")

st.markdown("#### Final Performance")

e, f, g, h = st.columns([2,2,2,3])
e.metric("Accuracy (Weighted Recall)", "0.6694", border=True)
f.metric("Weighted F1-score", "0.6660", border=True)
g.metric("Weighted Precision", "0.6660", border=True)


col13, col14, col15 = st.columns(3)

with col14:
    st.markdown("**Classification Report**")

    df_report = pd.DataFrame(
        [
            [0.69, 0.76, 0.72],
            [0.63, 0.55, 0.59]
        ],
        columns=["Precision", "Recall", "F1-score"],
        index=["Normal (0)", "Risk (1)"]
    )

    st.dataframe(
        df_report.style.format("{:.2f}").background_gradient(cmap="Blues", vmin=0.5, vmax=1.0),
        use_container_width=True
    )
    
with col13:
    cm_data = [[28715, 9266], 
            [12603, 15567]]

    x_labels = ["Predicted Normal", "Predicted Risk"]
    y_labels = ["Actual Normal", "Actual Risk"]

    fig = px.imshow(
        cm_data,
        x=x_labels,
        y=y_labels,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto"
    )

    fig.update_layout(
        title="Confusion Matrix Heatmap",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)