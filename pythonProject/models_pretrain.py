import torchvision

# train_data = torchvision.datasets.ImageNet("./dataset", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_flase = torchvision.models.vgg16(pretrained=False)
vgg16_ture = torchvision.models.vgg16(pretrained=True)#输出为1000
#print(vgg16_ture)

#cifar输出是10 任务是改变vgg16输出
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor())

#方法1 加一层线性变换 1000 到 10

vgg16_ture.classifier.add_module("add_linear", nn.Linear(1000,10))
print(vgg16_ture)

#方法2 改变最后一层的线性变换输出1000 to 10
vgg16_flase.classifier[6] = nn.Linear(4096, 10)
print(vgg16_flase)