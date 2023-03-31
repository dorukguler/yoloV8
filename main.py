import uvicorn
from fastapi import FastAPI
from routes.predict import predict_router

app = FastAPI()
app.include_router(predict_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



























# import base64
# with open("/Users/doruk/PycharmProjects/yoloV8/forklıft.jpeg", "rb") as img_file:
#     my_string = base64.b64encode(img_file.read())
# print(my_string)