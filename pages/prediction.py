import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

# 모델 불러오기
diabetes_model = joblib.load("model.pkl")
st.markdown("")
col21, col22, col23 = st.columns([1, 2, 1]) 

with col22:

    st.markdown("## ✅ Check Your Diabetes Risk")

    st.markdown("#### Input Your Health Checkup Information")
    with st.container(border=True):
        col24, col25, col26 = st.columns(3)

        with col24:
            st.markdown("**기본 정보**")
            gender = st.selectbox("성별", options=["남자", "여자"])
            age = st.selectbox("연령대", options=["20~24세", "25~29세", "30~34세", "35~39세","40~44세", "45~49세", "50~54세", "55~59세",
                    "60~64세", "65~69세", "70~74세", "75~79세","80~84세", "85세 이상"])
            height = st.number_input("신장 (cm)", min_value=100.0, max_value=250.0, value=165.0)
            weight = st.number_input("체중 (kg)", min_value=30.0, max_value=200.0, value=65.0)
            waist = st.number_input("허리둘레 (cm)", min_value=40, max_value=150, value=80)
            st.markdown("")
            st.markdown("**생활 습관**")
            smoking = st.selectbox("흡연 상태", options=["비흡연", "과거 흡연", "현재 흡연"])
            drinking = st.selectbox("음주 여부", options=["마시지 않음", "마심"])

        with col25:
            st.markdown("**혈압**")
            systolic_bp = st.number_input("수축기 혈압 (mmHg)", min_value=60, max_value=260, value=120)
            diastolic_bp = st.number_input("이완기 혈압 (mmHg)", min_value=20, max_value=170, value=80)
            st.markdown("")
            st.markdown("**혈액 검사**")
            total_cholesterol = st.number_input("총콜레스테롤 (mg/dL)", min_value=40, max_value=700, value=180)
            triglycerides = st.number_input("트리글리세라이드 (mg/dL)", min_value=20, max_value=600, value=150)
            hdl = st.number_input("HDL 콜레스테롤 (mg/dL)", min_value=10, max_value=150, value=50)
            ldl = st.number_input("LDL 콜레스테롤 (mg/dL)", min_value=10, max_value=300, value=100)
            hemoglobin = st.number_input("혈색소 (g/dL)", min_value=5.0, max_value=25.0, value=15.0)

        with col26:
            st.markdown("**신장 기능**")
            urine_protein = st.selectbox("요단백", options=["-", "±", "+1", "+2", "+3", "+4"])
            serum_creatinine = st.number_input("혈청 크레아티닌 (mg/dL)", min_value=0.1, max_value=5.0, value=1.0)
            st.markdown("")
            st.markdown("**간 기능**")
            ast = st.number_input("혈청지오티(AST) (IU/L)", min_value=0, max_value=300, value=20)
            alt = st.number_input("혈청지피티(ALT) (IU/L)", min_value=0, max_value=300, value=20)
            gamma_gtp = st.number_input("감마지피티 (IU/L)", min_value=0, max_value=500, value=20)


        # 입력받은 값 변환
        gender_code = 1 if gender == "남자" else 2
        bmi = weight / ((height / 100) ** 2)
        age_code = {
            "20~24세": 5,
            "25~29세": 6,
            "30~34세": 7,
            "35~39세": 8,
            "40~44세": 9,
            "45~49세": 10,
            "50~54세": 11,
            "55~59세": 12,
            "60~64세": 13,
            "65~69세": 14,
            "70~74세": 15,
            "75~79세": 16,
            "80~84세": 17,
            "85세 이상": 18
        }

        height_model = (height // 5) * 5
        weight_model = (weight // 5) * 5
        smoking_code = {
            "비흡연": 1,
            "과거 흡연": 2,
            "현재 흡연": 3
        }

        drinking_code = {
            "마시지 않음": 1,
            "마심": 2
        }

        urine_protein_code = {
            "-": 1,
            "±": 2,
            "+1": 3,
            "+2": 4,
            "+3": 5,
            "+4": 6
        }

        # 입력값 데이터프레임
        input_df = pd.DataFrame({
            "성별코드": [gender_code],
            "연령대코드(5세단위)": [age_code[age]],
            "신장(5cm단위)": [height_model],
            "체중(5kg단위)": [weight_model],
            "허리둘레": [waist],
            "수축기혈압": [systolic_bp],
            "이완기혈압": [diastolic_bp],
            "총콜레스테롤": [total_cholesterol],
            "트리글리세라이드": [triglycerides],
            "HDL콜레스테롤": [hdl],
            "LDL콜레스테롤": [ldl],
            "혈색소": [hemoglobin],
            "요단백": [urine_protein_code[urine_protein]],
            "혈청크레아티닌": [serum_creatinine],
            "혈청지오티(AST)": [ast],
            "혈청지피티(ALT)": [alt],
            "감마지티피": [gamma_gtp],
            "흡연상태": [smoking_code[smoking]],
            "음주여부": [drinking_code[drinking]],
            "BMI": [bmi]
        })

    # 예측

    if st.button("👉 **Check Risk**"):

        # 모델 예측
        prediction = diabetes_model.predict(input_df)[0]
        probability = diabetes_model.predict_proba(input_df)[0][1]

        # SHAP
        explainer = shap.TreeExplainer(diabetes_model)
        shap_values = explainer(input_df)

        importance = pd.DataFrame({
            "feature": input_df.columns,
            "shap": shap_values.values[0]
        })

        if prediction == 1:
            # 위험도를 높인 방향
            importance = importance.sort_values(
                "shap",
                ascending=False
            )
        else:
            # 위험도를 낮춘 방향
            importance = importance.sort_values(
                "shap",
                ascending=True
            )

        top3 = importance.head(3)

        # High Risk
        if prediction == 1:

            st.error("## High Risk Group")

            risk = probability * 100
            donut_color = "#ff4b4b"

            fig = go.Figure(
                data=[
                    go.Pie(
                        values=[risk, 100-risk],
                        hole=0.75,
                        sort=False,
                        direction="clockwise",
                        rotation=0,
                        textinfo="none",
                        marker=dict(
                            colors=[donut_color, "#eeeeee"]
                        )
                    )
                ]
            )

            fig.update_layout(
                showlegend=False,
                width=270,
                height=270,
                margin=dict(t=20, b=20, l=20, r=20),
                annotations=[
                    dict(
                        text=f"<b>{risk:.1f}%</b>",
                        x=0.5,
                        y=0.54,
                        showarrow=False,
                        font=dict(
                            size=28,
                            color=donut_color
                        )
                    ),
                    dict(
                        text="<span style='font-size:14px'>당뇨병 위험도</span>",
                        x=0.5,
                        y=0.38,
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=False)

            st.write("""
    현재 입력된 건강검진 정보를 바탕으로 **당뇨병 위험군**으로 예측되었습니다.

    생활습관 관리와 정기적인 건강검진, 필요 시 전문의 상담을 권장합니다.

    ※ 본 결과는 머신러닝 모델의 예측 결과이며 의학적 진단을 대체하지 않습니다.
    """)

        # Low Risk
        else:

            st.success("## Low Risk Group")

            risk = probability * 100
            donut_color = "#2E8B57"

            fig = go.Figure(
                data=[
                    go.Pie(
                        values=[risk, 100-risk],
                        hole=0.75,
                        sort=False,
                        direction="clockwise",
                        rotation=0,
                        textinfo="none",
                        marker=dict(
                            colors=[donut_color, "#eeeeee"]
                        )
                    )
                ]
            )

            fig.update_layout(
                showlegend=False,
                width=270,
                height=270,
                margin=dict(t=20, b=20, l=20, r=20),
                annotations=[
                    dict(
                        text=f"<b>{risk:.1f}%</b>",
                        x=0.5,
                        y=0.54,
                        showarrow=False,
                        font=dict(
                            size=28,
                            color=donut_color
                        )
                    ),
                    dict(
                        text="<span style='font-size:14px'>당뇨병 위험도</span>",
                        x=0.5,
                        y=0.38,
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=False)

            st.write("""
    현재 입력된 건강검진 정보를 바탕으로 **당뇨병 저위험군**으로 예측되었습니다.

    건강한 생활습관을 유지하시고, 정기적인 건강검진을 권장합니다.

    ※ 본 결과는 머신러닝 모델의 예측 결과이며 의학적 진단을 대체하지 않습니다.
    """)

        FEATURE_RECOMMENDATIONS = {
            "BMI": ["적정 체중(BMI) 유지","식후 15~20분 가벼운 산책"],
            "체중(5kg단위)": ["적정 체중 유지를 위한 식단 관리"],
            "허리둘레": ["복부 지방 감소를 위한 유산소 운동","정제 탄수화물 섭취 줄이기"],
            "수축기혈압": ["저염식 실천","규칙적인 유산소 운동"],
            "이완기혈압": ["저염식 실천","충분한 수면 및 스트레스 관리"],
            "총콜레스테롤": ["포화지방 섭취 줄이기","식이섬유 섭취 늘리기"],
            "트리글리세라이드": ["당류 및 음주 줄이기","규칙적인 유산소 운동"],
            "HDL콜레스테롤": ["규칙적인 유산소 운동","견과류 및 등푸른생선 섭취"],
            "LDL콜레스테롤": ["포화지방 섭취 줄이기","식이섬유 섭취 늘리기"],
            "흡연상태": ["금연 실천"],
            "음주여부": ["절주 및 금주 실천"],
            "혈청지오티(AST)": ["절주 및 금주 실천"],
            "혈청지피티(ALT)": ["절주 및 금주 실천"],
            "감마지티피": ["절주 및 금주 실천"]
        }

        DEFAULT_RECOMMENDATIONS = ["당분이 많은 음식 섭취 줄이기","정기적인 건강검진 받기"]

        with st.expander("예측에 가장 큰 영향을 준 건강 요인 (SHAP)"):

            if prediction == 1:
                st.markdown("#### Risk를 높이는 데 가장 크게 기여한 요인")
            else:
                st.markdown("#### Low Risk 예측에 가장 크게 기여한 요인")

            for i, row in enumerate(top3.itertuples(), 1):
                st.markdown(
                    f"**{i}. {row.feature}** "
                    f"(SHAP = {abs(row.shap):.3f})"
                )


        with st.expander("생활습관 개선 권장사항"):

            recommendations = []

            # SHAP Top3 중 권장 가능한 변수만 사용
            for feature in top3["feature"]:

                if feature in FEATURE_RECOMMENDATIONS:
                    recommendations.extend(
                        FEATURE_RECOMMENDATIONS[feature]
                    )

            # 공통 권장사항 추가
            recommendations.extend(DEFAULT_RECOMMENDATIONS)

            # 중복 제거
            recommendations = list(dict.fromkeys(recommendations))

            if prediction == 1:

                st.error("### 생활습관 개선 권장사항")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                st.write(
                    "위 권장사항은 이번 예측에 영향을 준 건강 요인을 "
                    "바탕으로 생성되었습니다."
                )

            else:

                st.success("### 건강한 생활습관 유지하기")
                for rec in recommendations:
                    st.markdown(f"- **{rec}**")
                st.write(
                    "현재 좋은 예측 결과에 기여한 건강 요인을 유지하는 것을 권장합니다."
                )
