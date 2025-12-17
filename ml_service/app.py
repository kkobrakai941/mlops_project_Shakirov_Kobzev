from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = FastAPI()

class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

@app.post("/predict")
async def predict(features: Features):
    data = [[features.feature1, features.feature2, features.feature3, features.feature4]]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
