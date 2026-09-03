import streamlit as st
import pandas as pd
import joblib

st.title("Previsao de Diabetes")

pipeline = joblib.load("modelo_diabetes.joblib")
pipeline_arvore = joblib.load("modelo_arvore.joblib")

pregnancies = st.number_input("Numero de gestacoes", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glicose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Pressao arterial", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Espessura da pele", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulina", min_value=0, max_value=900, value=80)
bmi = st.number_input("IMC", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.number_input("Historico familiar (DiabetesPedigreeFunction)", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.number_input("Idade", min_value=1, max_value=120, value=30)

if st.button("Prever"):
    entrada = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }])

    pred = pipeline.predict(entrada)[0]
    proba = pipeline.predict_proba(entrada)[0][1]

    st.write("\n============== REGRESSÃO LOGÍSTICA ==============")

    if pred == 1:
        st.write("Resultado: diabetico")
    else:
        st.write("Resultado: nao diabetico")

    st.write(f"Probabilidade de diabetes: {(round(proba, 3) * 100):.2f}" )

    #-----------------------------------------

    pred_arvore = pipeline_arvore.predict(entrada)[0]
    proba_arvore = pipeline_arvore.predict_proba(entrada)[0][1]

    st.write("\n============== ÁRVORE DE DECISÃO ==============")

    if pred_arvore == 1:
        st.write("Resultado: diabetico")
    else:
        st.write("Resultado: nao diabetico")

    # st.write(f"Probabilidade de diabetes (ÁRVORE DE DECISÃO): {(round(proba_arvore, 3) * 100):.2f}")
    st.write(f"Probabilidade de diabetes: {(round(proba_arvore, 3) * 100):.2f}%")