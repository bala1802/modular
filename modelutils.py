import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR

import yaml
from yaml.loader import SafeLoader

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

with open("modular/params.yaml") as f:
    params = yaml.load(f, Loader=SafeLoader)

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

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def construct_scheduler(optimizer, data_loader, epochs, maximum_learning_rate):
    scheduler = OneCycleLR(
        optimizer,
        max_lr=maximum_learning_rate,
        steps_per_epoch=len(data_loader),
        epochs=epochs,
        pct_start=params["one_cycle_lr_pct_start"]/epochs,
        # pct_start=params["one_cycle_lr_pct_start"],
        div_factor=params["one_cycle_lr_div_factor"],
        three_phase=False,
        final_div_factor=params["one_cycle_lr_final_div_factor"],
        anneal_strategy=params["one_cycle_lr_anneal_strategy"],
        verbose=params["one_cycle_lr_verbose"],
        
    )
    return scheduler