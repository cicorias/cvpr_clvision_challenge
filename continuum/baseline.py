import os
from typing import Iterable, Set, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

from continuum import ClassIncremental
from continuum.datasets import Core50
from continuum.tasks import split_train_val
from torchvision.transforms.transforms import Normalize, ToTensor

print(os.getcwd())

# Load the core50 data
core50 = Core50("../core50/data/", train=True, download=False)
core50_val = Core50("../core50/data", train=False, download=False)

# A new classes scenario
scenario = ClassIncremental(
    core50,
    increment=5,
    initial_increment=10,
    transformations=[ ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
)
scenario_val = ClassIncremental(
    core50_val,
    increment=5,
    initial_increment=10,
    transformations=[ ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
)

print(f"Number of classes: {scenario.nb_classes}.")
print(f"Number of tasks: {scenario.nb_tasks}.")

# Define a model
classifier = models.resnet18(pretrained=True)
classifier.fc = nn.Linear(512, 50)

if torch.cuda.is_available():
    print('cuda IS available')
    classifier.cuda()
else:
    print('cuda / GPU not available.')

# Tune the model hyperparameters
epochs = 1
lr = 0.01

# Define a loss function and criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=lr)

# Iterate through our NC scenario
for task_id, train_taskset in enumerate(scenario):
    # train_taskset, val_taskset = split_train_val(train_taskset, val_split=0.1)
    train_loader = DataLoader(train_taskset, batch_size=32, shuffle=True)
    
    unq_cls_train = np.unique(train_taskset._y)
    print(f"This task contains {len(unq_cls_train)} unique classes")
    print(f"Train: {unq_cls_train}")

    for epoch in range(epochs):

        print(f"<----- Epoch {epoch + 1} ------->")

        running_loss = 0.0
        train_total = 0.0
        train_correct = 0.0 
        for i, (x, y, t) in enumerate(train_loader):
            
            # Outputs batches of data, one scenario at a time
            x, y = x.cuda(), y.cuda()
            outputs = classifier(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print training statistics
            running_loss += loss.item()
            train_total += y.size(0)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == y).sum().item()
            
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                train_total = 0.0
                train_correct = 0.0            

    print("Finished Training")
    classifier.eval()

    # Validation against separate validation data
    for val_task_id, val_taskset in enumerate(scenario_val):
        if val_task_id > task_id:
            break

        val_loader = DataLoader(val_taskset, batch_size=32, shuffle=True)

        # Make sure we're validating the correct classes
        unq_cls_validate = np.unique(val_taskset._y)
        print(f"Validate: {unq_cls_validate}")

        total = 0.0
        correct = 0.0
        with torch.no_grad():
            for x, y, t in val_loader:
                x, y = x.cuda(), y.cuda()
                outputs = classifier(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        print(f"Validation Accuracy: {100.0 * correct / total}")
    
    classifier.train()

    