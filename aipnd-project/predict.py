import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import helper
from collections import OrderedDict

from PIL import Image

import os, random
import numpy as np

import json

import argparse

def get_command_line_args():
    parser = argparse.ArgumentParser(description="predict") #생성

    #각 args 옵션 명령에 따른 선택 파싱
    parser.add_argument("checkpoint", action="store", default="checkpoint.pth")
    parser.add_argument("--top_k", default="5")
    parser.add_argument("--category_names", default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", default=True)
    #action을 store_true로 하면 해당 옵션이 지정되면 True를 대입하라는 의미이다.

    return parser.parse_args()

def predict(image_path, model, topk):
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
    args = parse_args()

    checkpoint = args.checkpoint

    state = torch.load(checkpoint)
    learning_rate = state["learning_rate"]

    model = getattr(torchvision.models, state["arch"])(pretrained=True)
    model.epochs = state["epochs"]
    model.classifier = state["classifier"]
    model.load_state_dict(state["state_dict"])
    model.optimizer = state["optimizer"]
    model.class_to_idx = state["class_to_idx"]

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    data_dir = "flowers"
    test_dir = data_dir + "/test"

    image_sub_folder = random.choice(os.listdir(test_dir))
    image_file_name = random.choice(os.listdir(test_dir + "/" + image_sub_folder))
    image_path = test_dir + "/" + image_sub_folder + "/" + image_file_name

    top_k = args.top_k

    probs, classes = predict(image_path, model, top_k)

    max_index = np.argmax(probs)
    max_prob = probs[max_index]
    label = classes[max_index]

    print("Predicted class: {}, probability: {:.4f}".format(cat_to_name[label], probs)
    print("*****")
    for i in range(len(probs)):
        print("class: {}, probability: {:.4f}".format(classes[i], probs[i]))

if __name__ == "__main__":
    main()
