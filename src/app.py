import os
import streamlit as st
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from vgg import vgg11_bn
from torchsummary import summary


# load vgg something from torchvision without weights
# load cifar10 weights
# build inference pipeline

model = vgg11_bn(pretrained=True)
# summary(model, input_size=(3, 32, 32))

# can download data
# if not os.path.exists("../data"):
   # os.path.makedirs("../data")
# dataset = CIFAR10(root="../data", train=True, download=True)


# st.title("First ML App")
# st.write("""

#    ## First steps of learning to deploy a ML app.
# """)

