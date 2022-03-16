from fastapi import FastAPI, File, UploadFile
from PIL import Image
import onnxruntime as ort 
import numpy as np
import io
import math


app = FastAPI()

ort_session = ort.InferenceSession('models/binary_classifier_3.onnx')
TRHESHOLD = 0.5

@app.get("/")
async def root():
    return {"message": "Hello World"}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    input = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
    ort_inputs = {
        "input": input
    }
    ort_output = ort_session.run(['output'], ort_inputs)[0]
    output = sigmoid(ort_output)
    return {
        "proba": output,
        "label": "3" if output > TRHESHOLD else "no 3"
    }