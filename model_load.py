import torch

from model_save import vgg16

# model=torch.load('./vgg16.pth',weights_only=False)
# print(model)
vgg16.load_state_dict(torch.load('./vgg16_2.pth'))
print(vgg16)