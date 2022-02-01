import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader


class MinstCNN(nn.Module):
    def __init__(self):
        super(MinstCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


def prepare_dataloaders(data_path, batch_size, num_workers):
    transform = Compose([ToTensor(), Normalize((0.5), (0.5)) ])
    trainset = get_minst_dataset(data_path, transform, train=True)
    trainloader = get_dataloader(trainset, batch_size, num_workers)
    testset = get_minst_dataset(data_path, transform, train=False)
    testloader = get_dataloader(testset, batch_size, num_workers)
    return trainloader, testloader

def get_minst_dataset(data_path, transform, train):
    return datasets.MNIST(data_path, train, transform, download=True)

def get_dataloader(dataset, batch_size, num_workers):
    shuffle = True
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)

def train(model, num_epochs, dataloader, device):
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


if __name__ == '__main__':
    net = MinstCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    trainloader, testloader = prepare_dataloaders(
        data_path='./data',
        batch_size=24,
        num_workers=2
    )
    
    num_epochs = 10
    loss_history = train(net, num_epochs, trainloader, device)
    plot_training_loss(loss_history)

    model_path = './minst_net.pth'
    torch.save(net.state_dict(), model_path)

    test(net, testloader, device)