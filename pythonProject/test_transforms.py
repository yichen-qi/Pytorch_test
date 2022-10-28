from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
#1.transforms如何使用
#2.为什么需要tensor数据类型

img_path = "dataset/train/ants/6743948_2b8c096dda.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")




tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img.shape)
writer.add_image("Tensor_img", tensor_img)

writer.close()