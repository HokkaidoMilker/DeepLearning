import torch
from torch import nn
import torch.nn.functional as F
class tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input+1
        return output

if __name__ == '__main__':
    tudui = tudui()
    x = torch.tensor(1.0)
    y = tudui(x)
    print(y)
    F.conv2d()
