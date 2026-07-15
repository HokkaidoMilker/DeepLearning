from PIL import Image
import torchvision
from torchvision import transforms
import torch
image_path = "photo/unnamed.png"
image=Image.open(image_path).convert('RGB')

transform=torchvision.transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
image=transform(image)
print(image.shape)

model=torch.load("tudui_train.pth", weights_only=False)
model = model.cpu()
image = torch.reshape(image, (1,3,32,32))

model.eval()
with torch.no_grad():
    output = model(image)
    print(output)

print(output.argmax(1))

