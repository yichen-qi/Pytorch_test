import torch
import torch.nn as nn
import torch.nn.functional as F

class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, input):
        output = input + 1
        return output

tudui = Tudui()

# x = torch.tensor(1)
# output = tudui(x)
# print(output)

torch.save(tudui, "tudui.pth")