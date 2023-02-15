from fastapi import APIRouter,Body,File,UploadFile
import models.yolo
from utils.converter import base64_to_img
import base64
from fastapi import FastAPI, Response
from starlette.responses import FileResponse
from utils.converter import img_to_base64
import cv2 as cv



# File to use POST operation with predict functionality
predict_router= APIRouter()


@predict_router.post("/base64/")
async def predict_from_client_data(item: str = Body(...)):
    try:
        encoded = base64.b64encode(base64.b64decode(item))
        if encoded.decode() == item:
            image= base64_to_img(item)
        prediction = models.yolo.predict(image)
        return {"result" : img_to_base64(prediction)}
    except:
        return "Requested data does not the required format"

@predict_router.post("/file")
async def create_upload_file(file: UploadFile):
    result = models.yolo.predict(file)
    img = cv.imread(result)
    with open(img, "wb") as f:
        f.write(img)

    return FileResponse(img)

