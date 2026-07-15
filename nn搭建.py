import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear,Sequential
from torch.utils.tensorboard import SummaryWriter

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1=Sequential(
        conv1 = Conv2d(3, 32, 5, padding=2),
        maxpool1 = MaxPool2d(2),
        conv2 = Conv2d(32, 32, 5, padding=2),
        maxpool2 = MaxPool2d(2),
        conv3 = Conv2d(32, 64, 5, padding=2),
        maxpool3 = MaxPool2d(2),
        flatten = Flatten(),
        linear1 = Linear(1024, 64),
        linear2 = Linear(64, 10),

        )



    def forward(self, x):
        X= self.model1(x)
        return x


tudui = Tudui()

input = torch.ones(64, 3, 32, 32)
output=tudui(input)
print(output)
writer=SummaryWriter("flow")
writer.add_graph(tudui, input)
writer.close()