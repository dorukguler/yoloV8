import base64
import numpy as np
import cv2 as cv

# DECODE ETMEYI UNUTMA
def img_to_base64(img):
    im_bytes = img.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64


def base64_to_img(data):
    img_data = data.encode()
    im_bytes = base64.b64decode(img_data)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv.imdecode(im_arr, flags=cv.IMREAD_COLOR)
    return img




