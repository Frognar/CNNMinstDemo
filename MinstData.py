from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import datasets


class MinstDataset():
    def __init__(self, data_path, batch_size, num_workers):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = Compose([ToTensor(), Normalize((0.5), (0.5))])

    def get_trainloder(self):
        return self.get_dataloader(
            datasets.MNIST(
                self.data_path,
                train=True,
                transform=self.transform,
                download=True
            )
        )

    def get_testloder(self):
        return self.get_dataloader(
            datasets.MNIST(
                self.data_path,
                train=False,
                transform=self.transform,
                download=True
            )
        )

    def get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )