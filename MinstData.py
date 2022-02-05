from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import datasets


class MinstDataset():
    def __init__(self, data_path, device, batch_size, num_workers):
        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = Compose([ToTensor(), Normalize((0.5), (0.5))])

    def get_trainloader(self):
        return self.get_dataloader(
            datasets.MNIST(
                self.data_path,
                train=True,
                transform=self.transform,
                download=True
            )
        )

    def get_testloader(self):
        return self.get_dataloader(
            datasets.MNIST(
                self.data_path,
                train=False,
                transform=self.transform,
                download=True
            )
        )

    def get_dataloader(self, dataset):
        return WrappedDataLoader(
            DataLoader(
                dataset,
                self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            ),
            self.device
        )


class WrappedDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        batches = iter(self.dataloader)
        for x, y in batches:
            yield x.to(self.device), y.to(self.device)
    