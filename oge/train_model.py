import argparse
import os
from argparse import Namespace
from typing import Tuple

import numpy as np
import smdebug.pytorch as smd
import torch
import torchvision
from torch import nn
from torch.optim import lr_scheduler
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models
from torchvision import transforms as T

MODEL_CHECKPOINT_NAME = "oge_resnet50.pt"
SPLITS = ["train", "val"]


def transform_data(mean, std):
    """"
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std) 
    ])


def create_data_loaders(
    train_batch_size: int,
    test_batch_size: int,
    transforms: T.Compose, 
    shuffle: bool
) -> Tuple[DataLoader, dict, list]:
    fashion_dataset = {
        x: datasets.ImageFolder(
            args.train if x == "train" else args.test,
            transforms
        )
        for x in SPLITS
    }
    dataloaders = {
        x: DataLoader(
            fashion_dataset[x], 
            batch_size=train_batch_size if x == "train" else test_batch_size, 
            shuffle=shuffle
        )
        for x in SPLITS
    }
    dataset_sizes = {x: len(fashion_dataset[x]) for x in SPLITS}
    class_names = fashion_dataset["train"].classes

    return dataloaders, dataset_sizes, class_names


def get_normalization_stats(dataset):
    """
    Why and How to normalize data – Object detection on image in PyTorch Part 1
    https://inside-machinelearning.com/en/why-and-how-to-normalize-data-object-detection-on-image-in-pytorch-part-1/
    """
    imgs = torch.stack([img for img, _ in dataset], dim=3)
    imgs_view = imgs.view(3, -1)
    mean = imgs_view.mean(dim=1)
    std = imgs_view.std(dim=1)

    return mean, std


def net(num_output_classes: int = 8):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_output_classes)

    return model


def test(model, test_loader: DataLoader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    hook.set_mode(smd.modes.EVAL)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = 100 * correct / len(test_loader.dataset)
    
    # Log test loss and accuracy
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_acc
        )
    )
    return test_loss, test_acc


def train(model, train_loader: DataLoader, criterion, optimizer, scheduler, args: Namespace):
    model.to(args.device)
    model.train()
    hook.set_mode(smd.modes.TRAIN)

    for epoch in range(args.n_epochs):
        epoch_loss = 0
        correct = 0
        for data, target in train_loader:
            data = data.to(args.device)
            target = target.to(args.device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            epoch_loss += loss.item() * data.size(0)
            
            loss.backward()
            optimizer.step()

            prediction = outputs.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
        # scheduler.step()    # TODO: Check
        
        # Log training loss and accuracy 
        print(
            f"Epoch: {epoch}: Loss {epoch_loss/len(train_loader.dataset)}, "
            f"Accuracy: {100 * (correct / len(train_loader.dataset))}%"
        )
    return None


def main(args):
    model = net(num_output_classes=args.num_classes)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=args.scheduler_step_size, 
        gamma=args.scheduler_gamma
    )

    mean = (0, 0, 0)
    std = (1, 1, 1)
    if args.normalize:
        train_dataset = datasets.ImageFolder(
            args.train,
            T.Compose([
                T.ToTensor(),
                T.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
            ])
        )
        mean, std = get_normalization_stats(train_dataset)


    dataloaders, *_ = create_data_loaders(
        args.train_batch_size,
        args.test_batch_size,
        transforms=transform_data(mean, std),
        shuffle=args.shuffle
    )
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    train(
        model,
        dataloaders["train"], 
        loss_criterion, 
        optimizer, scheduler, 
        args
    )
    
    test(model, dataloaders["val"], loss_criterion)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, MODEL_CHECKPOINT_NAME)
    torch.save(model.state_dict(), model_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # ----
    # Data
    # ----
    parser.add_argument("--train", default=os.getenv("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--test", default=os.getenv("SM_CHANNEL_TEST", None))
    parser.add_argument("--num_classes", type=int, default=8, help="Number of classes in dataset")
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--shuffle", type=bool, default=True)

    # ---------
    # Modelling
    # ---------
    parser.add_argument("--train_batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument("--test_batch_size", type=int, default=64, help="input batch size for testing")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="") # TODO
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("-scheduler_step_size", type=int, default=2.0)
    parser.add_argument("--output_dir", default=os.getenv("SM_OUTPUT_DATA_DIR", None))
    parser.add_argument("--model_dir", default=os.getenv("SM_MODEL_DIR", None))
    
    # ---
    # GPU
    # ---
    parser.add_argument("--no_cuda", type=bool, default=False)
    
    args=parser.parse_args()
    
    args.device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    
    # Log all parameters (including hyperparameters)
    print(args)
    
    main(args)
