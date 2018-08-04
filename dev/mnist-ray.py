from __future__ import print_function
import argparse

import os
import shutil
from datetime import datetime

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
parser.add_argument('--validation-split', dest="val_split", type=float, default=0.2, 
                    help="size for validation set (default: 0.2)")
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--download', type=bool, default=True, 
                    help='disables download data')
parser.add_argument('--data', type=str, default="data/", metavar='PATH', 
                    help="root path for folder containing training data (default: data/")
parser.add_argument('--checkpoint', type=str, default='checkpoint/', metavar='PATH',
                        help='root path for folder containing model checkpoints \
                        (default: checkpoint/)')
args = parser.parse_args()

# average score to report
# FIXME: need to update this
def metric_average(val, name):
    tensor = torch.FloatTensor([val])
    return tensor.data[0]

def prepare_data(data=args.data, download=args.download, **kwargs):
    """Prepare Kaggle version of MNIST dataset with optional validation split"""
    train_dataset = datasets.MNIST(root=data, train=True, download=download,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    validate_dateset = datasets.MNIST(root=data, train=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    test_dataset = datasets.MNIST(root=data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    # get "masks" in order to split the dataset into subsets
    # From https://github.com/jvmancuso/cle-mnist/blob/master/prepare_data.py
    num_train = len(train_dataset)
    indices = np.arange(num_train)
    mask = np.random.sample(num_train) < args.val_split

    other_ix = indices[~mask]
    other_mask = np.random.sample(np.sum(~mask)) < (1-args.val_split)
    
    train_ix = other_ix[other_mask]
    val_ix = indices[mask]
    test_ix = np.arange(len(test_dataset))
    
    train_sampler = SubsetRandomSampler(train_ix)
    val_sampler = SubsetRandomSampler(val_ix)
    test_sampler = SubsetRandomSampler(test_ix)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(validate_dateset, batch_size=args.test_batch_size, sampler=val_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

    return train_loader, val_loader, test_loader

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_checkpoint(state, is_best, title, filename='checkpoint.pth.tar'):
    filepath = title + '-' + filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, title + '-best.pth.tar')

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
    
    # reproducibility
    # need to seed numpy/torch random number generators
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # need to figure out if downloaded data before
    if not os.path.isdir(args.data):
        mkdir_p(args.data)
    
    if os.path.exists(args.data + "processed/training.pt") and os.path.exists(args.data + "processed/test.pt"):
        args.download=False

    # need directory with checkpoint files to recover previously trained models
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    
    checkpoint_file = args.checkpoint + "net" + str(datetime.now())[:-10]
    
    # decide which device to use; assumes at most one GPU is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu") 

    # prep data loaders
    train_loader, val_loader, test_loader = prepare_data()

    # build model
    model = Net().to(device)
    kwargs = {'num_workers': 2} # specifies number of subprocesses to use for loading data
    
    # build optimizer
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

     # setup validation metrics we want to track for tracking best model over training run
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(1, args.epochs + 1):        
        print('\n================== TRAINING ==================')
        model.train() # set model to training mode

        # set up training metrics we want to track
        correct = 0
        train_num = len(train_loader.sampler)
        
        for batch_idx, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()

            pred = output.max(1, keepdim=True)[1] # get the index of the max logit
            correct += pred.eq(label.view_as(pred)).sum().item() # add to running total of hits

            if batch_idx % args.log_interval == 0: # maybe log current metrics to terminal
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(img), train_num,
                    100. * batch_idx / len(train_loader), loss.item()))

        # print whole epoch's training accuracy; useful for monitoring overfitting
        print('Train Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, train_num, 100. * correct / train_num))

        print('\n================== VALIDATION ==================')
        model.eval() # set model to evaluate mode

        # set up validation metrics we want to track
        val_loss = 0.
        val_correct = 0
        val_num = len(val_loader.sampler)

        # disable autograd here (replaces volatile flag from v0.3.1 and earlier)
        with torch.no_grad():
            # loop over validation batches
            for img, label in val_loader:
                img, label = img.to(device), label.to(device) # get data, send to gpu if needed
                output = model(img) # forward pass

                # sum up batch loss
                val_loss += F.cross_entropy(output, label, size_average=False).item()

                # monitor for accuracy
                pred = output.max(1, keepdim=True)[1] # get the index of the max logit
                val_correct += pred.eq(label.view_as(pred)).sum().item() # add to total hits

        # update current evaluation metrics
        val_loss /= val_num
        val_acc = 100. * val_correct / val_num
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, val_correct, val_num, val_acc))

        # check if best model according to accuracy;
        # if so, replace best metrics
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss # note this is val_loss of best model w.r.t. accuracy,
                                        # not the best val_loss throughout training

        # create checkpoint dictionary and save it;
        # if is_best, copy the file over to the file containing best model for this run
        state = {
            'epoch': epoch,
            'model': "Net",
            'state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc
        }

        save_checkpoint(state, is_best, checkpoint_file)    

        print('\n================== TESTING ==================')
        # load best model from training run (according to validation accuracy)
        check = torch.load(checkpoint_file + '-best.pth.tar')
        model.load_state_dict(check['state_dict'])
        model.eval() # set model to evaluate mode

        # set up evaluation metrics we want to track
        test_loss = 0.
        test_correct = 0
        test_num = len(test_loader.sampler)

        # disable autograd here (replaces volatile flag from v0.3.1 and earlier)
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                output = model(img)
                # sum up batch loss
                test_loss += F.cross_entropy(output, label, size_average=False).item()
                pred = output.max(1, keepdim=True)[1] # get the index of the max logit
                test_correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= test_num
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, test_num,
            100. * test_correct / test_num))

        print('Final model stored at "{}".'.format(checkpoint_file + '-best.pth.tar'))
        

tune.register_trainable("train_func", train)


all_trials = tune.run_experiments({
    "awesome": {
        "run": "train_func",
        "stop": {"mean_accuracy": 9},
        "config": {
            "lr": tune.grid_search(list(uniform.rvs(0, size=2))),
            "momentum": tune.grid_search(list(uniform.rvs(0, size=2))),
        }
    }
})

