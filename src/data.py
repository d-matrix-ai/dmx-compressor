import torch
import torchvision as tv
import numpy as np
from functools import partial


class MNIST():
    def __init__(self,
                 data_dir='~/data/mnist',
                 train_batch_size=64,
                 test_batch_size=1000,
                 cuda=False,
                 num_workers=1,
                 shuffle=True
                ):
        cuda_conf = {
            'num_workers': num_workers,
            'pin_memory': True,
        } if cuda else {}
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.train = torch.utils.data.DataLoader(
            tv.datasets.MNIST(
                data_dir, train=True, download=True, transform=transform),
            batch_size=train_batch_size,
            shuffle=shuffle,
            **cuda_conf)
        self.test = torch.utils.data.DataLoader(
            tv.datasets.MNIST(data_dir, train=False, transform=transform),
            batch_size=test_batch_size,
            shuffle=False,
            **cuda_conf)


class CIFAR():
    def __init__(self,
                 num_classes=10,
                 data_dir='~/data/cifar',
                 train_batch_size=64,
                 test_batch_size=64,
                 cuda=False,
                 num_workers=4,
                 shuffle=True):
        gpu_conf = {
            'num_workers': num_workers,
            'pin_memory': True
        } if cuda else {}
        # normalize = tv.transforms.Normalize(
        #     np.array([125.3, 123.0, 113.9]) / 255.0,
        #     np.array([63.0, 62.1, 66.7]) / 255.0)
        normalize = tv.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        transform_train = tv.transforms.Compose([
            tv.transforms.Pad(4, padding_mode='reflect'),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32),
            tv.transforms.ToTensor(), normalize
        ])
        transform_test = tv.transforms.Compose(
            [tv.transforms.ToTensor(), normalize])
        self.train = torch.utils.data.DataLoader(
            eval("tv.datasets.CIFAR{:d}".format(num_classes))(
                data_dir, train=True, download=True,
                transform=transform_train),
            batch_size=batch_size,
            shuffle=shuffle,
            **gpu_conf)
        self.test = torch.utils.data.DataLoader(
            eval("tv.datasets.CIFAR{:d}".format(num_classes))(
                data_dir, train=False, download=True,
                transform=transform_test),
            batch_size=batch_size,
            shuffle=False,
            **gpu_conf)


# aliases
CIFAR10 = partial(CIFAR, num_classes=10)
CIFAR100 = partial(CIFAR, num_classes=100)


class I1K():
    def __init__(self,
                 data_dir='~/data/I1K/i1k-extracted',
                 cuda=False,
                 num_workers=4,
                 batch_size=32,
                 shuffle=True):
        gpu_conf = {
            'num_workers': num_workers,
            'pin_memory': True
        } if cuda else {}
        normalize = tv.transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.train = torch.utils.data.DataLoader(
            tv.datasets.ImageFolder(
                data_dir + '/train',
                tv.transforms.Compose([
                    tv.transforms.RandomResizedCrop(224),
                    tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size,
            shuffle=shuffle,
            **gpu_conf)
        self.val = torch.utils.data.DataLoader(
            tv.datasets.ImageFolder(
                data_dir + '/validation',
                tv.transforms.Compose([
                    tv.transforms.Resize(256),
                    tv.transforms.CenterCrop(224),
                    tv.transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size,
            shuffle=False,
            **gpu_conf)


if __name__ == "__main__":
    pass