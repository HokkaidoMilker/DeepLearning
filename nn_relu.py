import torchvision
from torch import nn
from torch.nn import ReLU,Sigmoid
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter




dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=False, transform=torchvision.transforms.ToTensor())

data_test = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


if __name__ == '__main__':
    tudui = Tudui()

    writer = SummaryWriter("./logs_relu")
    step = 0
    for data in data_test:
        imgs, targets = data
        writer.add_images("input", imgs, global_step=step)
        output = tudui(imgs)
        writer.add_images("output", output, global_step=step)
        step += 1
        if step >= 2:
            break

    writer.close()