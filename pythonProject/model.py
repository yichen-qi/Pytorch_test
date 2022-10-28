from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
import torch

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

if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)

