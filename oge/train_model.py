#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import argparse
import os
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import lr_scheduler
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models
from torchvision import transforms as T


# TODO: Import dependencies for Debugging andd Profiling

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
    shuffle: bool, 
    num_workers: int
) -> Tuple[DataLoader, dict, list]:
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
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
            shuffle=shuffle, 
            num_workers=num_workers
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
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_output_classes)

    return model


def test(model, test_loader: DataLoader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()

    test_loss = running_loss / len(test_loader)
    test_acc = correct / len(test_loader)
    
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_acc
        )
    )
    return test_loss, test_acc


def train(model, train_loader: DataLoader, criterion, optimizer, scheduler, args: Namespace):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()

    for epoch in args.n_epochs:
        epoch_loss = 0
        correct = 0
        for data, target in train_loader:
            data = data.to(args.device)
            target = target.to(args.device)

            outputs = model(data)
            loss = criterion(outputs, target)
            epoch_loss += loss
            
            loss.backward()
            optimizer.step()

            prediction = outputs.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
        
        scheduler.step()    # TODO: Check
        print(
            f"Epoch: {epoch}: Loss {epoch_loss/len(train_loader.dataset)}, "
            f"Accuracy: {100 * (correct / len(train_loader.dataset))}%"
        )
    return None


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net(num_output_classes=args.num_classes)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # TODO: Include scheduler in training loop
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
        shuffle=args.shuffle,
        num_workers=-1,
    )
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model = train(
        model,
        dataloaders["train"], 
        loss_criterion, 
        optimizer, scheduler, 
        args
    )
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, dataloaders["test"], loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    model_path = os.path.join(args.model_dir, MODEL_CHECKPOINT_NAME)
    torch.save(model.state_dict(), model_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # ----
    # Data
    # ----
    parser.add_argument("--train", default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", default=os.environ["SM_CHANNEL_TEST"])
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
    parser.add_argument("--scheduler_gamma", type=float, default=0.0)
    parser.add_argument("-scheduler_step_size", type=int, default=2.0)
    parser.add_argument("--output_dir", default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", default=os.environ["SM_MODEL_DIR"])
    
    # ---
    # GPU
    # ---
    parser.add_argument("--no_cuda", action="store_true")
    args=parser.parse_args()
    
    args.device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    main(args)
