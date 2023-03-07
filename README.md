# Streamlit ML App

You can download model weights, download the code, run docker-compose to build and run the backend and frontend with the following steps:

## Download model weights

Place the image in a `state_dicts` folder under project root directory.

[Model weights](https://drive.google.com/file/d/1-ExlM56fxIL3cfIVpp-rckUXEAXBLt2j/view?usp=share_link) to Google drive for weights of simple VGG model trained on CIFAR10 dataset.

## Install locally

```bash
git clone git@github.com:jonasdieker/image_classification_ml_app.git
cd image_classification_ml_app
docker-compose up -d --build
```

Then go to [http://localhost:8501/](http://localhost:8501/) to interact with the web app and classify some images!