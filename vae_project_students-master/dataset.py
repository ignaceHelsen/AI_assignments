import torch
import torchvision
import torchvision.transforms as transforms

import wandb
from denormalize import Denormalize


class Dataset:
    def __init__(self, config: dict):
        """
        Note: config.dataset should contain either MNIST or CIFAR10.
        """
        self.config = config

        if config["dataset"] == "CIFAR10":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            trainset = torchvision.datasets.CIFAR10('datasets/', train=True, download=True,
                                                    transform=transform)
            testset = torchvision.datasets.CIFAR10('datasets/', train=False, download=True,
                                                   transform=transform)
            self.denormalize = Denormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
            trainset = torchvision.datasets.MNIST('datasets/', train=True, download=True,
                                                  transform=transform)
            testset = torchvision.datasets.MNIST('datasets/', train=False, download=True,
                                                 transform=transform)
            self.denormalize = Denormalize((0.1307,), (0.3081,))

        if config["dataset"] not in ["MNIST", "CIFAR10"]:
            print("Dataset not correctly defined, loading the MNIST digits dataset...")

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config["batch_size"], shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=config["batch_size_test"], shuffle=True)

    def show_examples(self):
        """
        Show some examples of the selected dataset.
        """
        wandb.init(project="VAE_project", entity="the-elite-schakels", name="Dataset_test")

        examples = enumerate(self.train_loader)
        _, (example_data, example_targets) = next(examples)

        print(f"The shape of the training data tensor is: {example_data.shape}")
        wandb.log(
            {
                "Dataset samples": wandb.Image(
                    self.denormalize(example_data)
                )
            }
        )
