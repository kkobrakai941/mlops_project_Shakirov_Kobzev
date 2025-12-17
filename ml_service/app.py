from fastapi import FastAPI, Response
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import os
import psycopg2
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# обучаем модель на наборе iris
iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

# параметры подключения к БД
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "mlops")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

# создаем таблицу predictions при пуске сервиса
with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                feature1 FLOAT,
                feature2 FLOAT,
                feature3 FLOAT,
                feature4 FLOAT,
                prediction INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

# метрики Prometheus
REQUEST_COUNT = Counter('prediction_requests_total', 'Total number of prediction requests')
PREDICTION_DISTRIBUTION = Histogram('prediction_values', 'Distribution of prediction values')


@app.post("/predict")
async def predict(features: Features):
    data = [[features.feature1, features.feature2, features.feature3, features.feature4]]
    pred = model.predict(data)[0]
    # сохранение результата в БД
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (feature1, feature2, feature3, feature4, prediction) VALUES (%s, %s, %s, %s, %s)",
                (features.feature1, features.feature2, features.feature3, features.feature4, int(pred))
            )
            conn.commit()
    # обновляем метрики
    REQUEST_COUNT.inc()
    PREDICTION_DISTRIBUTION.observe(pred)
    return {"prediction": int(pred)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type='text/plain')
