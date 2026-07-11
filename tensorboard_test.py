from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import PIL


image_path1=r"data_set/val/ants/153320619_2aeb5fa0ee.jpg"
image1=PIL.Image.open(image_path1)
image_ndarry1=np.array(image1)


image_path2=r"data_set/val/ants/540543309_ddbb193ee5.jpg"
image2=PIL.Image.open(image_path2)
image_ndarry2=np.array(image2)


summary = SummaryWriter("logs")
summary.add_image("ants",image_ndarry1,1,dataformats="HWC")
summary.add_image("ants",image_ndarry2,2,dataformats="HWC")


for i in range(100):

    summary.add_scalar("y=2*x", 2*i, i)
summary.close()
