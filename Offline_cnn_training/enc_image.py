import cv2 
import tenseal as ts
from torchvision import transforms

def encrypt_img(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = transform(image)
    return image