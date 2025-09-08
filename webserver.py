from fastapi import FastAPI
import joblib
from pydantic import BaseModel

model = joblib.load("regression.joblib")

app = FastAPI()


class HouseFeatures(BaseModel):
    size: float
    nb_rooms: int
    garden: bool

@app.post("/predict")
async def predict(features: HouseFeatures):
    prediction = model.predict([[features.size, features.nb_rooms, features.garden]])[0]
    return {"prediction": prediction}