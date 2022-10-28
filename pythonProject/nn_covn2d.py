import torch
import torchvision
import torch.nn as nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)


dataloader = DataLoader(dataset=test_data, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1,padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()

writer = SummaryWriter("dataloader")
print(tudui)
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images("input", imgs, step)

    output = torch.reshape(output, (-1,3,30,30))
    writer.add_images("output", output, step)
    step = step + 1
