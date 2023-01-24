import uvicorn

import ultralytics
from ultralytics import YOLO
import cv2 as cv
import numpy
import torch
import os
from fastapi import File,FastAPI
import base64
from pydantic import BaseModel
import io
from PIL import Image


# main.py
from fastapi import FastAPI
# from models import yolo_model
from routes.predict import predict_router

app = FastAPI()
app.include_router(predict_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)













# cv.imwrite(predicted_path,img)
#
# cv.imshow("Predicted Image",img)
#
#
# cv.waitKey(0)

# def img_to_base64(file):
#     with open(file,"rb") as image_file:
#         return  base64.b64encode(image_file.read()).decode("utf-8")



























# import base64
# with open("/Users/doruk/PycharmProjects/yoloV8/forklıft.jpeg", "rb") as img_file:
#     my_string = base64.b64encode(img_file.read())
# print(my_string)