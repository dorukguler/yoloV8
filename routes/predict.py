import cv2 as cv
# routes/predict.py
from fastapi import APIRouter,Body
from pydantic import BaseModel

import models.yolo
from utils.converter import base64_to_img
from utils.converter import img_to_base64

import base64


# File to use POST operation with predict functionality

predict_router = APIRouter()


@predict_router.post("/predict")
async def predict_from_client_data(item: str = Body(...)):
    # Decode the base64 image string into bytes
    # image_bytes = image.decode()
    # print(item)
    try:
        encoded = base64.b64encode(base64.b64decode(item))
        if encoded.decode() == item:
            image= base64_to_img(item)
        prediction = models.yolo.predict(image)
        #     return img_to_base64(prediction)
        return prediction
    except:
        return "Requested data does not the required format"
    # Use your YOLO model to predict on the image
    # prediction = predict(image_bytes)
    return {"result" : predict_from_client_data}
