import os
import streamlit as st
import torch
from torchvision.datasets import CIFAR10
from vgg import vgg11_bn

from utils import load_and_prep_image, class_names, predict_json, update_logger


# can download data
# if not os.path.exists("../data"):
   # os.path.makedirs("../data")
# dataset = CIFAR10(root="../data", train=True, download=True)

google_credentials = ""
PROJECT = ""
REGION = ""


def get_model():
   """load vgg model pre-trained on CIFAR10."""
   return vgg11_bn(pretrained=True)


def print_model_arch():
   model = get_model()
   return summary(model, input_size=(3, 32, 32))


def make_predicition(image, model, class_names):
   """Takes image from session_state, loads it and predicts its class."""

   # switch model to eval mode
   model.eval()
   assert model.training == False, "Model is not in 'eval' mode!"

   # loading the image
   image = load_and_prep_image(image)

   # preds = predict_json(project=PROJECT,
   #                      region=REGION,
   #                      model=model,
   #                      instances=image)

   # pred_class = class_names[preds[1]]
   # pred_conf = max(preds[0])
   # return pred_class, pred_conf
   
   # predicting class
   softmax = torch.nn.Softmax(dim=1)
   prediction = softmax(model(image))
   pred_class = torch.argmax(prediction, dim=1)
   return class_names[pred_class.item()], float(prediction[0, [pred_class.item()]])


def main():
   st.title("First ML App")
   st.header("Classify what's in your photos!")

   if st.checkbox("Show supported classes"):
      st.write(f"These are the classes the model can identify\n: {class_names}")

   # upload file
   uploaded_file = st.file_uploader(label="Upload image", type=["png", "jpeg", "jpg"])

   if not uploaded_file:
      st.warning("Please upload an image.")
      st.stop()
   else:
      st.session_state["uploaded_image"] = uploaded_file
      st.image(st.session_state["uploaded_image"], use_column_width=True)
      pred_button = st.button("Predict", key="pred_button")

   if st.session_state.pred_button:
      model = get_model()
      st.session_state.pred_class, st.session_state.pred_conf = make_predicition(uploaded_file, model, class_names)
      st.write(f"Prediction: {st.session_state.pred_class},\
         Confidence: {st.session_state.pred_conf:.3f}")


if __name__ == "__main__":
   main()
