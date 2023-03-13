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

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(img,(x1,y1),(x2,y2),(255,0,0),5)
    # cv.rectangle(img, (int(bbox_arr[1][0]), int(bbox_arr[1][1])), (int(bbox_arr[1][2]), int(bbox_arr[1][3])),
    #              color=(255, 0, 0), thickness=4)
    return img
