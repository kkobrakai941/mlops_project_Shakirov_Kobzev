import streamlit as st
import requests
import pandas as pd
import psycopg2
import os

st.title("Сервис прогнозирования оттока клиентов")

feature1 = st.number_input("Признак 1", value=0.0)
feature2 = st.number_input("Признак 2", value=0.0)
feature3 = st.number_input("Признак 3", value=0.0)
feature4 = st.number_input("Признак 4", value=0.0)

if st.button("Предсказать"):
    payload = {"feature1": feature1, "feature2": feature2, "feature3": feature3, "feature4": feature4}
    try:
        resp = requests.post("http://ml_service:8000/predict", json=payload)
        if resp.status_code == 200:
            prediction = resp.json().get("prediction")
            st.success(f"Предсказанный класс: {prediction}")
        else:
            st.error("Ошибка запроса к ML сервису")
    except Exception:
        st.error("Ошибка соединения с ML сервисом")

db_host = os.getenv("DB_HOST","db")
db_port = os.getenv("DB_PORT","5432")
db_name = os.getenv("DB_NAME","mlops_db")
db_user = os.getenv("DB_USER","postgres")
db_password = os.getenv("DB_PASSWORD","postgres")

try:
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    df = pd.read_sql("SELECT id, feature1, feature2, feature3, feature4, prediction, created_at FROM predictions ORDER BY created_at LIMIT 100", conn)
    st.subheader("История предсказаний")
    st.dataframe(df)
    conn.close()
except Exception:
    st.write("История предсказаний недоступна")
