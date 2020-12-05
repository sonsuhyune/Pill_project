import torch
import torchvision.transforms.functional as FT
import cv2
from PIL import Image
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def contrast_processing(image):
    """
    Histogram equalization
    :param img:image, a PIL Image
    :return:image, a PIL Image
    """
    scale = 300
    img = np.asarray(image)

    blur_img = cv2.GaussianBlur(image, (0, 0), scale / 30)
    merge_img = cv2.addWeighted(image, 4, blur_img, -4, 128)
    merge_img = merge_img.astype('uint8')

    merge_img = cv2.cvtColor(merge_img, cv2.COLOR_BGR2RGB)
    contrast_img = FT.to_pil_image(merge_img)
    return contrast_img

def resize(image, dims=(300, 300)):
    """
    Resize
    :param img:image, a PIL Image
    :return:image, a PIL Image
    """
    # Resize image
    resized_image = FT.resize(image, dims)

    return resized_image

def crop(image, coordinates):
    coordinates = int(coordinates)
    xmin, ymin, xmax, ymax = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    cropped_img = image.crop((xmin, ymin, xmax, ymax))
    return cropped_img

def preprocess_SSD(img):
    # resize for SSD
    img = resize(img, dims=(300, 300))
    # Convert PIL image to Torch tensor
    img = FT.to_tensor(img)
    # normalize
    img = FT.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    return img

def preprocess_SSD(img):
    # resize for SSD
    img = resize(img, dims=(300, 300))
    # Convert PIL image to Torch tensor
    img = FT.to_tensor(img)
    # normalize
    img = FT.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    return img

def preprocess_EAST(img, pill_boxes):
    # crop
    img = crop(img, pill_boxes)
    # resize for SSD
    img = resize(img, dims=(256, 256))
    # Convert PIL image to Torch tensor
    img = FT.to_tensor(img)
    # normalize
    img = FT.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    return img

def preprocess_CRNN(img, text_boxes):
    # crop
    img = crop(img, text_boxes)
    # enhance contrast
    img = contrast_processing(img)
    # resize for SSD
    img = resize(img, dims=(256, 256))
    # Convert PIL image to Torch tensor
    img = FT.to_tensor(img)
    # normalize
    img = FT.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    return img