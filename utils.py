import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder

def construct_train_loader(train):
    dataloader_args = dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    return train_loader

def construct_test_loader(test):
    dataloader_args = dict(shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return test_loader

def construct_optimizer(model, learning_rate, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                           weight_decay=weight_decay)
    return optimizer

def construct_cross_entropy_loss():
    return nn.CrossEntropyLoss()

def construct_LR_finder(model, optimizer, criterion, device, dataloader, 
                        end_learning_rate, number_of_iterations, step_mode):
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(dataloader, end_lr=end_learning_rate, 
                         num_iter=number_of_iterations, step_mode=step_mode)
    # lr_finder.plot()
    # lr_finder.reset()
    return lr_finder