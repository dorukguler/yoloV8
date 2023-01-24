import ultralytics
from ultralytics import YOLO
import cv2 as cv


# Load the model
def load_weights():
    weights = r"/Users/doruk/PycharmProjects/yoloV8/best3.pt"
    return weights

def predict(img):
    model = YOLO(load_weights())
    results = model.predict(source=img)
    # Draw Bboxes
    for result in results:
        # convert tensor array to numpy
        bbox_arr = result.boxes.xyxy.numpy()
        cv.rectangle(img, (int(bbox_arr[1][0]), int(bbox_arr[1][1])), (int(bbox_arr[1][2]), int(bbox_arr[1][3])),
                     color=(255, 0, 0), thickness=4)  # ,cv.COLORR[int(bbox_arr[5])], 3)
        # cv.rectangle(path,((int(bbox_arr[0]),int(bbox_arr[1]))

