import cv2 as cv
from ultralytics import YOLO
import base64
from utils.converter import base64_to_img
import numpy as np
#
# data = input()
# # # decoded = base64.b64decode(data)
# img_data = data.encode()
# # # encoded = base64.b64encode(base64.b64decode(data))
# # # # print(encoded.decode() == data)
# # #
# # # bytes = base64_to_img(data)
# # # print(bytes)
# im_bytes = base64.b64decode(img_data)
# im_arr = np.frombuffer(im_bytes, dtype=np.uint8) # im_arr is one-dim Numpy array
# img = cv.imdecode(im_arr, flags=cv.IMREAD_COLOR)
# #
# model = YOLO("/Users/doruk/PycharmProjects/yoloV8/models/best3.pt")
# results = model.predict(img)
#
# for result in results:
#     print(result.boxes.xyxy.numpy())
#
#
# # [[        455          49         774         330]
# #  [          1          61         445         492]
# #  [        756          81        1851         600]
# #  [        186          80         443         481]]

