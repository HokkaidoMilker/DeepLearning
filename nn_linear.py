import torchvision
from torch import nn
from torch.nn import Linear
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=False, transform=torchvision.transforms.ToTensor())

data_test = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        input = torch.flatten(input, start_dim=1)   # 把 64×3×32×32 展平成 64×3072
        output = self.linear1(input)
        return output


if __name__ == '__main__':
    tudui = Tudui()

    writer = SummaryWriter("./logs_linear")
    step = 0
    for data in data_test:
        imgs, targets = data
        writer.add_images("input", imgs, global_step=step)
        output = tudui(imgs)
        print(f"batch {step}: input shape={imgs.shape}, output shape={output.shape}")
        step += 1
        if step >= 2:
            break

    writer.close()