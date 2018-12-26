import torch

from torch import optim
from torch.utils.data import DataLoader

from digitrecognizer.dataset import Mnist, mnist_dataset
from digitrecognizer.network import Net
import torch.nn as nn


def data(file, test_ratio):
    trainset, testset = mnist_dataset(file, test_ratio)

    trainloader = DataLoader(trainset)

    testloader = DataLoader(testset)

    return trainloader, testloader


if __name__ == '__main__':
    trainloader, testloader = data('input/train.csv', 0.1)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
