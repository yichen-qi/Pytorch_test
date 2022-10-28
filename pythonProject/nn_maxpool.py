#torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, # return_indices=False, ceil_mode=False)

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)



class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output

tudui = Tudui()
step = 0
writer = SummaryWriter("dataloader")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs,step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1
    print(imgs.shape)
    print(output.shape)

writer.close()
