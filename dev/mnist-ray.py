from __future__ import print_function
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import ray
import ray.tune as tune

from scipy.stats import uniform

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device("cuda")
    torch.cuda.manual_seed(args.seed)

if args.cuda:
    # Move model to GPU.
    model.cuda()


# average score to report
# FIXME: need to update this
def metric_average(val, name):
    tensor = torch.FloatTensor([val])
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.data[0]

def prepare_data():
    """Prepare Kaggle version of MNIST dataset with optional validation split"""
    train_dataset = datasets.MNIST("data", train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    test_dataset = datasets.MNIST("data", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    # get "masks" in order to split the dataset into subsets
    # From https://github.com/jvmancuso/cle-mnist/blob/master/prepare_data.py
    num_train = len(train_dataset)
    indices = np.arange(num_train)
    mask = np.random.sample(num_train) < args.test_split
    other_ix = indices[~mask]
    other_mask = np.random.sample(np.sum(~mask)) < args.train_split
    train_ix = other_ix[other_mask]
    test_ix = indices[mask]
    if not np.all(other_mask):
        val_ix = other_ix[~other_mask]
    else:
        val_ix = None


    train_sampler = SubsetRandomSampler(train_ix)

    test_sampler = SubsetRandomSampler(test_ix)

    return train_sampler, test_sampler, train_dataset, test_dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Update this to use ray
ray.init()
          
def train(config, reporter):
    
    model = Net()

    train_sampler, test_sampler, train_dataset, test_dataset = prepare_data()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

    for epoch in range(1, args.epochs + 1):
        model = ConvNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                    momentum=config["momentum"])

        model.train()
        
        for idx, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.data[0]))

                model.eval()
                test_loss = 0.
                test_accuracy = 0.

                for data, target in test_loader:
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    data, target = Variable(data, volatile=True), Variable(target)
                    output = model(data)
                    # sum up batch loss
                    test_loss += F.nll_loss(output, target, size_average=False).data[0]
                    # get the index of the max log-probability
                    pred = output.data.max(1, keepdim=True)[1]
                    test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

                test_loss /= len(test_sampler)
                test_accuracy /= len(test_sampler)

                test_loss = metric_average(test_loss, 'avg_loss')
                test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

                reporter(timesteps_total=idx,
                    mean_accuracy = test_accuracy)


tune.register_trainable("train_func", train)


all_trials = tune.run_experiments({
    "awesome": {
        "run": "train_func",
        "stop": {"mean_accuracy": 9},
        "config": {
            "lr": tune.grid_search(list(uniform.rvs(0, size=20))),
            "momentum": tune.grid_search(list(uniform.rvs(0, size=20))),
        }
    }
})

