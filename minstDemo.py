import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from datetime import datetime

from MinstModel import MinstCNN
from MinstData import MinstDataset


def parse_argument():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None, help='Path to checkpoint')
    ap.add_argument('-o', '--output', default='./', help='output directory')
    return vars(ap.parse_args())

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f'Checkpoint path {checkpoint_path} does not exist')
    return torch.load(checkpoint_path)

def save_checkpoint(out_dir, epoch, loss, accuracy, model, optimizer):
    create_dir_if_not_exists(out_dir)
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    torch.save({
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        f'{out_dir}/checkpoint_e{epoch}_{now}.pth'
    )

def save_model(model, out_dir):
    create_dir_if_not_exists(out_dir)
    torch.save(model.state_dict(), f'{out_dir}/model.pth')

def train(model, dataloader, loss_func, optimizer):
    model.train()
    epoch_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss

def test(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

def plot_training_history(epoch, losses, accuracies):
    epochs = [i for i in range(1, epoch + 1)]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='r')
    ax1.plot(epochs, losses, color='r', label='Loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='b')
    ax2.plot(epochs, accuracies, color='b', label='Accuracy')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = parse_argument()
    checkpoint_path = args['checkpoint']
    out_dir = args['output']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MinstDataset('./data', device, batch_size=24, num_workers=2)
    trainloader = dataset.get_trainloader()
    testloader = dataset.get_testloader()

    model = MinstCNN()
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    last_epoch = 0
    losses = []
    accuracies = []
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path)
        last_epoch = checkpoint['epoch']
        losses = checkpoint['loss']
        accuracies = checkpoint['accuracy']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    epochs = 10
    end_epoch = last_epoch + epochs
    for epoch in range(last_epoch + 1, end_epoch + 1):
        loss = train(model, trainloader, loss_func, optimizer)
        accuracy = test(model, testloader)
        last_epoch = epoch
        losses.append(loss)
        accuracies.append(accuracy)
        save_checkpoint(out_dir, epoch, losses, accuracies, model, optimizer)
        print(f'[{epoch}/{end_epoch}] loss: {loss:.4f} accuracy: {accuracy}')

    plot_training_history(last_epoch, losses, accuracies)
    save_model(model, out_dir)