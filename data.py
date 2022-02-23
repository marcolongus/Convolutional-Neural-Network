import torch
import torchvision
from torchvision import transforms
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt


class Data():

    train = datasets.FashionMNIST(
        "FMINST data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test = datasets.FashionMNIST(
        "FMINST data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    label = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot",
        }

    def __init__(self, label=label):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label = label

    def train(self, batch_size=10, train=train):
        return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    def test(self, batch_size=10, test=test):
        return torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    def train_set(self, batch_size=1000):
        train_set = [
            (image.view(-1, 1, 28, 28).to(self.device), target.to(self.device))
            for image, target in self.train(batch_size)
        ]
        return train_set

    def test_set(self, batch_size=100):
        test_set = [
            (image.view(-1, 1, 28, 28).to(self.device), target.to(self.device))
            for image, target in self.test(batch_size)
        ]
        return test_set


if __name__ == "__main__":

    data = Data()
    train_set = data.train_set(2)
    test_set = data.test_set(2)

    for element in test_set:
        print(element[1])
        break

    for batch in train_set:
        images, target = batch
        print(type(images), type(target))
        for element in target:
            print(element)


        print(images.shape, target.shape)
        for i, image in enumerate(images):
            plt.title(data.label[target[i].cpu().item()])
            plt.imshow(image.cpu().view(28, 28))
            plt.show()
        break
