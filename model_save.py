import  torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=True)

torch.save(vgg16, './vgg16.pth')

torch.save(vgg16.state_dict(), './vgg16_2.pth')