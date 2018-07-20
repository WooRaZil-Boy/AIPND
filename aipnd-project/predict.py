import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable

from collections import OrderedDict

from PIL import Image

import os, random
import numpy as np
import json

import argparse

def get_command_line_args():
    parser = argparse.ArgumentParser(description="train") #생성

    #각 args 옵션 명령에 따른 선택 파싱
    parser.add_argument("input", action="store")
    parser.add_argument("checkpoint", action="store", default="checkpoint.pth")
    parser.add_argument("--top_k", default=5)
    parser.add_argument("--category_names", default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", default=True)
    #action을 store_true로 하면 해당 옵션이 지정되면 True를 대입하라는 의미이다.

    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    size = 224
    width, height = image.size

    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)

    resized_image = image.resize((width, height)) #pillow의 resize

    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size

    cropped_image = image.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))

    return np_image_array

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval() #평가
    model = model.cuda() #GPU

    image = Image.open(image_path) #PIL의 Image 객체
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array) #np 배열을 torch 텐서로 변환

    inputs = Variable(tensor.float().cuda())

    inputs = inputs.unsqueeze(0)
    output = model.forward(inputs)

    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu()
    classes = ps[1].cpu()
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()

    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])

    return probabilities.numpy()[0], mapped_classes

def main():
    args = get_command_line_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    state = torch.load(args.checkpoint)
    learning_rate = state["learning_rate"]

    model = getattr(torchvision.models, state["arch"])(pretrained=True)
    model.epochs = state["epochs"]
    model.classifier = state["classifier"]
    model.load_state_dict(state["state_dict"])
    model.optimizer = state["optimizer"]
    model.class_to_idx = state["class_to_idx"]

    predicted_probability, predicted_class = predict(args.input, model, args.top_k)

    print("Predicted Classes: ", predicted_class)
    print("Class Names: ")
    print([cat_to_name[x] for x in predicted_class])
    print("Predicted Probability: ", predicted_probability)

if __name__ == "__main__":
    main()
