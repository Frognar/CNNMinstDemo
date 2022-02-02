import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from datetime import datetime

from MinstModel import MinstCNN
from MinstData import MinstDataset


def train(model, num_epochs, dataloader, device, output_dir):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_history = {}
    for epoch in range(num_epochs):
        losses = 0.0
        for (images, labels) in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()
        print(f'[{epoch + 1}/{num_epochs}] loss: {losses / len(dataloader)}')
        loss_history[epoch] = losses
        now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        model_path = f'{output_dir}/minstModel_l{losses}_{now}.pth'
        torch.save(model.state_dict(), model_path)
    return loss_history

def plot_training_loss(loss_history):
    epochs, losses = loss_history.keys(), loss_history.values()
    plt.plot(epochs, losses, 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def test(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy of the model on the test images: {accuracy:.1f}')

def parse_argument():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', help='path to saved model')
    ap.add_argument('-o', '--output', help='output directory', default='.')
    return vars(ap.parse_args())

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_model(model_path=None):
    model = MinstCNN()
    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Loaded model from {model_path}')
    return model

if __name__ == '__main__':
    args = parse_argument()
    output_dir = args['output']

    create_dir_if_not_exists(output_dir)
    net = get_model(args['model'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    dataset = MinstDataset(
        data_path='./data',
        batch_size=24,
        num_workers=2
    )
    trainloader = dataset.get_trainloder()
    
    num_epochs = 10
    loss_history = train(net, num_epochs, trainloader, device, output_dir)
    plot_training_loss(loss_history)

    model_path = f'{output_dir}/minst_net.pth'
    torch.save(net.state_dict(), model_path)

    testloader = dataset.get_testloder()
    test(net, testloader, device)