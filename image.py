import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn


def make_model(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.eval().to(device)
    return model


def rgb_to_gray(image):
    gray_image = image.convert('L')
    gray_image_rgb = gray_image.convert('RGB')
    return gray_image_rgb


def image(images_path, device, pred_len):
    img = Image.open(images_path)
    img = rgb_to_gray(img)
    preprocess = transforms.ToTensor()
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    ResNet = make_model(device)

    with torch.no_grad():
        image_feature = ResNet(img_tensor).flatten()

    linear_layer = nn.Linear(image_feature.shape[0], pred_len).to(device)  # 线性层移到设备

    with torch.no_grad():
        image_feature = linear_layer(image_feature)

    image_feature = image_feature.detach().cpu().numpy()

    return image_feature
