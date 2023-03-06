from fastapi import APIRouter,Body,File,UploadFile
import models.yolo
from utils.converter import base64_to_img
import base64
from fastapi import FastAPI, HTTPException, Response
from starlette.responses import FileResponse
from utils.converter import img_to_base64
import cv2 as cv
from typing import Dict, Any



# File to use POST operation with predict functionality
predict_router= APIRouter()


@predict_router.post("/base64/")
async def predict_from_client_data(item: str = Body(...)) -> Dict[str, Any]:
    """
    Checks whether the input is a valid Base64 encoded string or not.
    If it is valid, then it decodes it to an image and makes a prediction on it using a YOLO model.
    If it is not valid, then it raises an HTTP 400 Bad Request error.
    """
    try:
        # First, check if the input is a valid Base64 encoded string
        encoded = base64.b64encode(base64.b64decode(item))
        if encoded.decode() != item:
            raise ValueError("Input is not a valid Base64 encoded string")

        # If the input is valid, then decode it to an image
        image = base64_to_img(item)

        # Make a prediction on the image using the YOLO model
        prediction = models.yolo.predict(image)

        # Convert the prediction back to a Base64 encoded image string
        result = img_to_base64(prediction)

        # return the result as a JSON object
        return {"result" : result}

    except ValueError as e:
        # If the input is not valid, raise an HTTP 400 Bad Request error
        raise HTTPException(status_code=400,detail="Invalid input format: must be a valid Base64 encoded string")

    except Exception as e:
        # If an unexpected error occurs, raise an HTTP 500 Internal Server Error
        raise HTTPException(status_code=500, detail=str(e))


@predict_router.post("/file")
async def create_upload_file(file: UploadFile):
    result = models.yolo.predict(file)
    img = cv.imread(result)
    with open(img, "wb") as f:
        f.write(img)

    return FileResponse(img)

