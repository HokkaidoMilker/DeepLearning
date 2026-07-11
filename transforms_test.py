from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

IMAGE_PATH=r"data_set/val/bees/54736755_c057723f64.jpg"
IMG=Image.open(IMAGE_PATH)
transform=transforms.ToTensor()
tensor=transform(IMG)

writer=SummaryWriter("log")
writer.add_image("image",tensor,0)
writer.close()
print(tensor)