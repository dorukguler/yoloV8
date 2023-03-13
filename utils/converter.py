import base64
import numpy as np
import cv2 as cv
from PIL import Image
from io import BytesIO

def img_to_base64(img):
    # img = cv.imread(img)
    _, im_arr = cv.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64


def base64_to_img(data):
    img_b64 = data.encode()
    im_bytes = base64.b64decode(img_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv.imdecode(im_arr, flags=cv.IMREAD_COLOR)
    return img



