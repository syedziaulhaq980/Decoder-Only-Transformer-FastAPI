from fastapi import FastAPI
from pydantic import BaseModel

from transformer_model import predict_next_token


app = FastAPI(
    title="Decoder-Only Transformer API",
    version="1.0.0"
)


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Decoder-Only Transformer API is running"
    }


@app.post("/predict")
def predict(data: TextInput):

    next_token = predict_next_token(data.text)

    return {
        "input_text": data.text,
        "next_token": next_token
    }
