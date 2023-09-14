import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose
import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision
from skimage import io
import argparse
from .custom_dataset_loader import StateFarmDataset


def load_data(batch_size, dataset_name, num_works, train_params, model_params):
    if dataset_name == 'MNIST':
        train_set, test_set = load_mnist_dataset(batch_size, num_works)
        model_params['hin'] = 28
        model_params['in_size'] = 1
        model_params['out_size'] = 10
        train_params['save_acc'] = True
    elif dataset_name == 'CIFAR10':
        train_set, test_set = load_cifar_dataset(batch_size, num_works)
        model_params['hin'] = 32
        model_params['in_size'] = 3
        model_params['out_size'] = 10
        train_params['save_acc'] = True
    elif dataset_name == 'STATEFARM':
        train_set, test_set = load_statefarm_dataset(batch_size, num_works)
        model_params['hin'] = 42   #640 on resnet
        model_params['in_size'] = 3
        model_params['out_size'] = 10
        train_params['save_acc'] = True
    else:
        raise ValueError('To implement')

    train_params['nb_batches'] = train_set.__len__()
    train_params['p'] = train_set.dataset.__len__()
    return train_set, test_set


def load_mnist_dataset(batch_size, num_works):
    dataset = MNIST('.', download=True, transform=transforms.ToTensor(), train=True)
    train_set = DataLoader(dataset, batch_size=batch_size, num_workers=num_works)
    dataset = MNIST('.', download=True, transform=transforms.ToTensor(), train=False)
    test_set = DataLoader(dataset, batch_size=batch_size, num_workers=num_works)
    return train_set, test_set


def load_cifar_dataset(batch_size, num_works):

    train_transform = Compose([transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomCrop(32, padding=4),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = CIFAR10('.', download=True, transform=train_transform, train=True)
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_works)

    test_set = CIFAR10('.', download=True, transform=test_transform, train=False)
    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_works)
    print('CIFAR LOADER')
    return train_set, test_set

def load_statefarm_dataset(batch_size, num_works):
    transform = transforms.Compose(
            [transforms.ToPILImage(),
                transforms.Resize(128),transforms.ToTensor()
             ])
    statefarm_dataset = StateFarmDataset(csv_file='/root/driver_imgs_list.csv', root_dir='/root/imgs/train',
                                                          transform=transform)
    train_length=int(0.7* len(statefarm_dataset))
    test_length=len(statefarm_dataset)-train_length
    train_dataset,test_dataset = torch.utils.data.random_split(statefarm_dataset,(train_length,test_length))
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_works)
    test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_works)
    return train_set, test_set
