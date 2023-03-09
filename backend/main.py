import uvicorn
from fastapi import FastAPI, File, Response, UploadFile
from PIL import Image

from backend.inference import get_prediction, load_and_prep_image

app = FastAPI()

# TODO: Make this into a class and load model in constructor, add get_pred as normal method, and two routes
# https://www.anyscale.com/blog/serving-pytorch-models-with-fastapi-and-ray-serve


@app.get("/")
def is_alive():
    print("/isalive requested")
    status_code = Response(status_code=200)
    return status_code


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image = Image.open(image.file)
    print("/predict requested")
    image = load_and_prep_image(image)
    pred_class, pred_conf = get_prediction(image)
    return {"pred_class": pred_class, "pred_conf": pred_conf}


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
