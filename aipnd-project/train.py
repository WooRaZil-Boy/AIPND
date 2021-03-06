import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

import argparse

def get_command_line_args():
    parser = argparse.ArgumentParser(description="train") #생성

    #각 args 옵션 명령에 따른 선택 파싱
    parser.add_argument("data_dir", action="store")
    parser.add_argument("--save_dir", action="store")
    parser.add_argument("--arch", default="vgg19", choices=["vgg13", "vgg19"])
    parser.add_argument("--learning_rate", default="0.001")
    parser.add_argument("--hidden_units", default=4096)
    parser.add_argument("--epochs", default="10")
    parser.add_argument("--gpu", action="store_true", default=True)
    #action을 store_true로 하면 해당 옵션이 지정되면 True를 대입하라는 의미이다.

    return parser.parse_args()

def train(model, learning_rate, hidden_units, epochs, dataloaders, image_datesets_size):
    for param in model.parameters():
        param.requires_grad = False

    input_size = 25088
    hidden_sizes = [hidden_units, int(hidden_units/2), int(hidden_units/4)]

    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(input_size, hidden_sizes[0])),
        ("relu1", nn.ReLU()),
        ("drop1", nn.Dropout(p=0.5)),
        ("fc2", nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ("relu2", nn.ReLU()),
        ("drop2", nn.Dropout(p=0.5)),
        ("fc3", nn.Linear(hidden_sizes[1], hidden_sizes[2])),
        ("output", nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    epochs = epochs
    print_every = 20

    model.to("cuda")

    for e in range(epochs):
        print("\nEpoch {} / {}".format(e + 1, epochs))

        for phase in ["training", "validation"]:
            if phase == "training":
                model.train(True)
            else:
                model.train(False)

            running_loss = 0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                optimizer.zero_grad()

                #Forward
                outputs = model.forward(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == "training":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            running_corrects += int(torch.sum(preds == labels.data))

        epoch_loss = running_loss / image_datesets_size[phase]
        epoch_acc = running_corrects / image_datesets_size[phase]

        print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

    return model, optimizer

def save(model, optimizer, arch, learning_rate, input_size, hidden_sizes, epochs):
    state = {
        "arch": arch,
        "learning_rate": learning_rate,
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "epochs": epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "classifier": model.classifier,
        "class_to_idx": model.class_to_idx
    }

    torch.save(state, "checkpoint.pth")

def main():
    args = get_command_line_args()

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    transforms_size = 256
    transforms_cropSize = 224
    transforms_means = [0.485, 0.456, 0.406]
    transforms_std = [0.229, 0.224, 0.225]

    data_transforms = {
        "training": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(transforms_cropSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(transforms_means,
                                transforms_std)
        ]),

        "validation": transforms.Compose([
            transforms.Resize(transforms_size),
            transforms.CenterCrop(transforms_cropSize),
            transforms.ToTensor(),
            transforms.Normalize(transforms_means,
                               transforms_std)
        ]),

        "testing": transforms.Compose([
            transforms.Resize(transforms_size),
            transforms.CenterCrop(transforms_cropSize),
            transforms.ToTensor(),
            transforms.Normalize(transforms_means,
                               transforms_std)
        ])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        "training": datasets.ImageFolder(train_dir, transform=data_transforms["training"]),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
        "testing": datasets.ImageFolder(test_dir, transform=data_transforms["testing"])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders =  {
        "training": torch.utils.data.DataLoader(image_datasets["training"], batch_size=64, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=64, shuffle=False),
        "testing": torch.utils.data.DataLoader(image_datasets["training"], batch_size=64, shuffle=False)
    }

    arch = args.arch
    learning_rate = float(args.learning_rate)
    hidden_units = int(args.hidden_units)
    epochs = int(args.epochs)

    model = getattr(torchvision.models, args.arch)(pretrained=True)

    image_datesets_size = {
        "training": len(image_datasets["training"]),
        "validation": len(image_datasets["validation"])
    }

    model, optimizer = train(model, learning_rate, hidden_units, epochs, dataloaders, image_datesets_size)

    model.class_to_idx = image_datasets["training"].class_to_idx
    save(model, optimizer, arch, learning_rate, 25088, hidden_units, epochs)

if __name__ == "__main__":
    main()
