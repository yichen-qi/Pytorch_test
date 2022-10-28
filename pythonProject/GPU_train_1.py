import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
# from model import *
#gpu cuda 加快网络 损失函数 数据（img target）


train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                         transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)
#lenth
train_data_size = len(train_data)
test_data_size = len(test_data)
#格式化字符串
print("train data length : {}".format(train_data_size))
print("test data length : {}".format(test_data_size))

#利用dataloader加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#构建神经网络

#方法1 直接写网络
class Tudui(nn.Module):
    def __init__(self):
            super(Tudui, self).__init__()
        # simply
            self.modell = Sequential(
                Conv2d(3, 32, 5, padding=2),
                MaxPool2d(2),
                Conv2d(32, 32, 5, padding=2),
                MaxPool2d(2),
                Conv2d(32, 64, 5, padding=2),
                MaxPool2d(2),
                Flatten(),
                Linear(1024, 64),
                Linear(64, 10)
              )
        #simply
    def forward(self, x):
            x = self.modell(x)
            return x

tudui = Tudui()
tudui = tudui.cuda() # grafic

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
#优化器
learning_rate = 1e-2
optim = torch.optim.SGD(tudui.parameters(), lr=learning_rate)#定义优化器

#设置训练网络的参数
total_train_step = 0   #记录训练次数
total_test_step = 0    #记录测试次数
epoch = 30             #训练轮数

writer = SummaryWriter("logs")
start_time = time.time()
for i in range(epoch):
    print("-------the {}th epoch-------".format(i+1))
    #训练开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optim.zero_grad()  # 清零
        loss.backward()  # 得到权重,梯度
        optim.step()  # 运行优化
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0: #减少输出，每一百次输出一次
            end_time = time.time()
            print("train time: {}".format(end_time - start_time))
            print("the {}th train, loss : {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs ,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum() #网络中算出的output最大值的位置和tearget相等的个数
            total_accuracy = total_accuracy + accuracy


    print("total test loss : {}".format(total_test_loss))
    print("total accuracy : {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    torch.save(tudui, "tudui_{}.pth".format(i+1))
    print("model save finished")



writer.close()