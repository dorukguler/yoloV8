import ultralytics
from ultralytics import YOLO
import cv2 as cv
import os
from utils.converter import img_to_base64


absolute_path = os.path.dirname(__file__)
relative_path = "best3.pt" # Your model weights
full_path = os.path.join(absolute_path, relative_path)


# Load the model weights
def load_weights():
    weights = f"{full_path}"
    return weights

def predict(img):
    model = YOLO(load_weights())
    results = model.predict(source=img)

    for result in results:
        bbox_arr = result.boxes.xyxy

    cv.rectangle(img, (int(bbox_arr[1][0]), int(bbox_arr[1][1])), (int(bbox_arr[1][2]), int(bbox_arr[1][3])),
                 color=(255, 0, 0), thickness=4)

    response_image = img_to_base64(img)
    return response_image

