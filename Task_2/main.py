import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import time
device = "cpu"
torch.set_num_threads(4)

batch_size = 256 # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    # remember to exit the train loop at end of the epoch
    
    # set model in training mode
    model.train()
    # initialize variables to keep track of beginning and end of iterations index 1-39
    start_time = end_time = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # standard training loop
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print the loss value after every 20 iterations
        if batch_idx % 20 == 0:
            print(f"Loss value after {batch_idx} iterations: {loss}")
        # record start time since first iteration just finished
        if batch_idx == 0:
            start_time = time.time()
        # record end time since 40th iteration just finished and print avg time per iteration
        if batch_idx == 39:
            end_time = time.time()
            print(f"Average time per iteration: {(end_time - start_time)/39}")
            break

    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main(args):

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    # begin distributed data parallel training with port number set to 8890
    torch.distributed.init_process_group('gloo', init_method=f"tcp://{args.master_ip}:8890",rank=args.rank, world_size=args.num_nodes)

    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    # create distributed sampler used for distributed data parallel training
    sampler = DistributedSampler(training_set)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    shuffle=False,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    # register model with distributed data parallel
    model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        sampler.set_epoch(epoch)
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    # use argparse to read in command line parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--master_ip')
    parser.add_argument('--num_nodes', type=int, default=3)
    parser.add_argument('--rank', type=int)

    args = parser.parse_args()

    # pass in arguments that were read in to main method
    main(args)
