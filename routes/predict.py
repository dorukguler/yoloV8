
# routes/predict.py
from fastapi import APIRouter

predict_router = APIRouter()


@predict_router.post("/predict")
async def predict(image: str):
    # Decode the base64 image string into bytes
    image_bytes = image.decode()

    # Use your YOLO model to predict on the image
    prediction = predict(image_bytes)
    return prediction