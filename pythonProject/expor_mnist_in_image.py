import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                         transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)
#lenth


train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

train_data_size = len(train_dataloader)
test_data_size = len(test_dataloader)

print(train_data_size)
print(test_data_size)

os.mkdir('images')

for i in range(test_data_size):
    data, target = next(iter(test_dataloader)) # 迭代器
    new_data = data[0][0].clone().numpy() # 拷贝数据
    plt.imsave('images/'+str(i)+str(target)+'.png', new_data)
    print(target)

