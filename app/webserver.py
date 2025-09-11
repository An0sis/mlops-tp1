from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from transformers import pipeline

model = joblib.load("app/regression.joblib")
pipe = pipeline("summarization", model="google-t5/t5-small")

app = FastAPI()


class HouseFeatures(BaseModel):
    size: float
    nb_rooms: int
    garden: bool
    
class SummarizationRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(features: HouseFeatures):
    prediction = model.predict([[features.size, features.nb_rooms, features.garden]])[0]
    return {"prediction": prediction}

@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    summary = pipe(request.text)
    return {"summary": summary[0]['summary_text']}

@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API!"}