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
    ap.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint'
    )
    ap.add_argument(
        '--model',
        type=str,
        default=None,
        help='path to saved model'
    )
    ap.add_argument(
        '--checkpoint_output',
        type=str,
        default='./checkpoints',
        help='checkpoint output directory'
    )
    ap.add_argument(
        '-o', '--output',
        type=str,
        default='./',
        help='output directory'
    )
    return vars(ap.parse_args())

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise ValueError(f'Checkpoint path {checkpoint_path} does not exist')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['last_epoch']
    loss_history = checkpoint['loss_history']
    return last_epoch, loss_history

def train(model, loss_func, optimizer, epochs,
        dataloader, checkpoint_dir, last_epoch, loss_history):
    model.train()
    end_epoch = last_epoch + epochs
    for epoch in range(last_epoch + 1, end_epoch + 1):
        loss = process_epoch(model, dataloader, loss_func, optimizer)
        loss_history[epoch] = loss
        save_checkpoint(checkpoint_dir, model, optimizer, epoch, loss_history)
        print(f'[{epoch}/{end_epoch}] loss: {loss:.4f}')

def process_epoch(model, dataloader, loss_func, optimizer):
    epoch_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss

def save_checkpoint(checkpoint_dir, model, optimizer, last_epoch, loss_history):
    torch.save({
        'last_epoch': last_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history
    }, name_checkpoint(checkpoint_dir, last_epoch))

def name_checkpoint(output_dir, last_epoch):
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    return f'{output_dir}/checkpoint_e{last_epoch}_{now}.pth'

def plot_training_loss(loss_history):
    epochs, losses = loss_history.keys(), loss_history.values()
    plt.plot(epochs, losses, 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

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
        print(f'Test Accuracy of the model on the test images: {accuracy:.1f}')

if __name__ == '__main__':
    args = parse_argument()
    checkpoint = args['checkpoint']
    model_path = args['model']
    checkpoint_output_dir = args['checkpoint_output']
    output_dir = args['output']
    create_dir_if_not_exists(output_dir)
    create_dir_if_not_exists(checkpoint_output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MinstDataset('./data', device, batch_size=24, num_workers=2)
    trainloader = dataset.get_trainloader()
    testloader = dataset.get_testloader()

    model = MinstCNN()
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    last_epoch = 0
    loss_history = {}

    if checkpoint is not None:
        last_epoch, loss_history = load_checkpoint(checkpoint, model, optimizer)

    epochs = 10
    train(model, loss_func, optimizer, epochs, trainloader,
        checkpoint_output_dir, last_epoch, loss_history)
    plot_training_loss(loss_history)
    save_model(model, f'{output_dir}/model.pth')
    test(model, testloader)