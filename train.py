import torch

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = outputs.argmax(dim=1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct /total

    return train_loss, train_accuracy

