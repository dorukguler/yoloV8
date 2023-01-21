import base64
import io
import json
from PIL import Image
from fastapi import File,FastAPI
# import torchuvicorn
# from main import get_predictions


# create the app
app = FastAPI(title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",)


# Define your paths & methods for your API

@app.get('/')
async def getapi():
    return {"message": "GET API test"}




# @app.post("/objectdetection/")
# async def get_body(file: bytes = File(...)):
#   return {"result": "ok"}

# @app.post("/objectdetection/")
# async def get_body(file: bytes = File(...)):
#   input_image =Image.open(io.BytesIO(file)).convert("RGB")
#   results = model(input_image)
#   results_json =   json.loads(results.pandas().xyxy[0].to_json(orient="records"))
#   return {"result": results_json}

# @app.post("/object-to-img")
# async def detect_forklift_return_base64_img(file: bytes = File(...)):
#     input_arr = get_predictions()
#     arr_to_base64 = base64.b64decode(input_arr)
#     for img in results.imgs:
#         bytes_io = io.BytesIO()
#         img_base64 = Image.fromarray(img)
#         img_base64.save(bytes_io, format="jpeg")
#     return arr_to_base64