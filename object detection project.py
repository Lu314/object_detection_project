import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms.functional import to_tensor
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8) # Corrected typo here
    model.eval()
    return model

model = load_model()

def make_prediction(img):
    img_tensor = img_preprocess(img)  # Use the global img_preprocess directly
    prediction = model([img_tensor])  # Make sure img_tensor is correctly formatted
    prediction = prediction[0]
    prediction["labels"] = [categories[label.item()] for label in prediction["labels"]]
    prediction["scores"] = prediction["scores"].detach().numpy()  # Detach for further processing
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)
    labels = [f"{label}: {score:.2f}" for label, score in zip(prediction["labels"], prediction["scores"])]  # scores are already detached
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=labels, colors=["red" if label.startswith("person") else "green" for label in prediction["labels"]], width=4)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np


# Dashboard
st.title("Object Detector :tea: :coffee:")
upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)
    prediction = make_prediction(img)  # Dictionary
    img_with_bbox = create_image_with_bboxes(img, prediction)

    fig, ax = plt.subplots(figsize=(12, 12))  # Corrected figure creation
    ax.imshow(img_with_bbox)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

    st.pyplot(fig)

    st.header("Predicted Probabilities")
    st.write(prediction)