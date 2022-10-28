import torchvision
from PIL import Image
from model import *

image_path = "D:\mnist_task\my_image\pi.PNG"
image = Image.open(image_path)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32))
                                            , torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

tudui = Tudui()
tudui = tudui.cuda()

model = torch.load("tudui_30.pth")
print(model)

image = torch.reshape(image, (-1,3,32,32))
image = image.cuda()
model.eval()

with torch.no_grad():
    output = model(image)

print(output)

print(output.argmax(1))