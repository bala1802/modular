from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, scheduler, epoch, loss_function):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0

    losses = []

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        y_pred = model(data)

        # loss = F.nll_loss(y_pred, target)
        loss = loss_function(y_pred, target)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step(sum(losses)/len(losses))

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        
    # scheduler.step(sum(losses)/len(losses))