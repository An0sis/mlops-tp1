from fastapi import FastAPI
import joblib

model = joblib.load("regression.joblib")

app = FastAPI()


@app.post("/predict")
async def predict(
    size: float, nb_rooms: int, garden: bool
):
    prediction = model.predict([[size, nb_rooms, garden]])[0]
    return {"prediction": prediction}