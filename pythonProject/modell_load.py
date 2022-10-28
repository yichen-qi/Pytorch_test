import torch
from nn_module import *
import torchvision.models


#方式1 加载模型
model = torch.load("vgg16_method1.pth")
print(model)

#方式2 加载数据
model = torch.load("vgg16_method2.pth")
print(model)

#方式2 加载的数据变成模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

#导入自己写的网络 先导入class

model = torch.load("tudui.pth")
print(model)