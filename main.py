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


from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.engine.predictor import BasePredictor


# Create the app
app = FastAPI(title="Custom YOLOV8 API",
    description="""Detect object from an image and return as base64 """,
    version="0.0.1",)


# Path to image to predict on
path = r"/Users/doruk/PycharmProjects/yoloV8/forklıft.jpeg"



# Save as numpy array with opencv
img = cv.imread(path)

# Load the model
model = YOLO("/Users/doruk/PycharmProjects/yoloV8/best3.pt")

# predict from ndarray with opencv
results = model.predict(source=img)    # save=True,save_txt=True


for result in results:
    # convert tensor array to numpy
    bbox_arr = result.boxes.xyxy.numpy()
    cv.rectangle(img, (int(bbox_arr[1][0]), int(bbox_arr[1][1])), (int(bbox_arr[1][2]), int(bbox_arr[1][3])),color=(255, 0, 0),thickness=4) #,cv.COLORR[int(bbox_arr[5])], 3)
    # cv.rectangle(path,((int(bbox_arr[0]),int(bbox_arr[1]))

# Path to predicted image
predicted_path = "/Users/doruk/PycharmProjects/yoloV8/ultralytics/runs/detect/predict"
cv.imwrite(predicted_path,img)
cv.imshow("Predicted Image",img)


cv.waitKey(0)

def img_to_base64(file):
    with open(file,"rb") as image_file:
        return  base64.b64encode(image_file.read()).decode("utf-8")





# @app.get('/')
# async def getapi():
#     return {"message": "GET API test"}

# @app.post("/predict")
# async def predict_on_image():
#     '''
#     FastAPI API will take an image as input and return it to base64
#     '''
#     results = model(input_image)
#     input_image = img_to_base64(predicted_path)
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
#     detect_res = json.loads(detect_res)
#     return {"result": detect_res}



#
# while True:
#     # load the model
#     model = YOLO("/Users/doruk/PycharmProjects/yoloV8/best3.pt")
#
#     # read the image and save to img variable
#     img  = cv.imread(path)
#
#     # Predict on image
#     results = model.predict(img)
#
#     cv.imwrite(,img)
#
#     for result in results:
#         # convert tensor array to numpy


#
#
# for result in results:
#     print(result.boxes.xyxy.numpy())










# def save(self, labels=True, save_dir='/Users/doruk/PycharmProjects/yoloV8/ultralytics/runs/detect/predict'):
#     save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/predict', mkdir=True)  # increment save_dir
#     self.display(save=True, labels=labels, save_dir=save_dir)  # save results




# import base64
# with open("/Users/doruk/PycharmProjects/yoloV8/forklıft.jpeg", "rb") as img_file:
#     my_string = base64.b64encode(img_file.read())
# print(my_string)