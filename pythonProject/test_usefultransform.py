from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants/6743948_2b8c096dda.jpg")
print(img)
# totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
trans_norm = transforms.Normalize([1, 3, 4],[3, 2, 1])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

# resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)





writer.close()