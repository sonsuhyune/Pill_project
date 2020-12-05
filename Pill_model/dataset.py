import glob
import re
import xml.etree.ElementTree as ET

#import cv2
from PIL import Image
import numpy as np

import torch
from torch.utils import data
import torchvision.transforms.functional as FT
import torch.nn.functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

'''
def contrast_processing(image):
    """
    Histogram equalization
    :param img:image, a PIL Image
    :return:image, a PIL Image
    """
    scale = 300
    image = np.asarray(image)

    blur_img = cv2.GaussianBlur(image, (0, 0), scale / 30)
    merge_img = cv2.addWeighted(image, 4, blur_img, -4, 128)
    merge_img = merge_img.astype('uint8')

    merge_img = cv2.cvtColor(merge_img, cv2.COLOR_BGR2RGB)

    contrast_img = FT.to_pil_image(merge_img)

    return contrast_img
'''

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
    """
    Crop
    :param image: PIL image
    :param coordinates: model result
    :return: list of crop boxes
    """
    cropped_img_list = []
    for i in range(len(coordinates)):
        if len(coordinates[i]) == 4: #ssd
            xmin, ymin, xmax, ymax = int(coordinates[i,0]), int(coordinates[i,1]), int(coordinates[i,2]), int(coordinates[i,3])
            cropped_img_list.append(image.crop((xmin, ymin, xmax, ymax)))
        elif len(coordinates[i]) == 8: #east
            xmin = int(min(coordinates[i, [0, 2, 4, 6]]))
            xmax = int(max(coordinates[i, [0, 2, 4, 6]]))
            ymin = int(min(coordinates[i, [1, 3, 5, 7]]))
            ymax = int(max(coordinates[i, [1, 3, 5, 7]]))
            cropped_img_list.append(image.crop((xmin, ymin, xmax, ymax)))
        else:
            print("Wrong format of coordinates")
    return cropped_img_list

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def preprocess_YOLO(img):
    """
    :param img: tensor
    :return:
    """
    img = FT.to_pil_image(img).convert('L').convert('RGB')
    img = FT.to_tensor(img)
    _, h, w = img.shape
    # Pad to square resolution
    img, pad = pad_to_square(img, 0)
    img = FT.to_pil_image(img)
    img = resize(img, dims=(416, 416))
    img = FT.to_tensor(img)
    img = img.unsqueeze(0)
    return img

def postprocess_YOLO(img, pill_boxes):
    # crop
    img_list = crop(img, pill_boxes)
    return img_list

def preprocess_SSD(img):
    # resize for SSD
    img = FT.to_pil_image(img)
    img = resize(img, dims=(300, 300))
    # Convert PIL image to Torch tensor
    img = FT.to_tensor(img)
    # normalize
    # img = FT.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    img = FT.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img = img.unsqueeze(0)
    return img

def postprocess_SSD(img, pill_boxes):
    # crop
    img_list = crop(img, pill_boxes)
    return img_list

def preprocess_EAST(pill_list):
    """
    :param pill_list: list of cropped pill image(PIL image)
    :return: resized batch format tensor
    """
    batch_img = []
    for i in range(len(pill_list)):
        # resize for SSD
        img = resize(pill_list[i], dims=(256, 256))
        # Convert PIL image to Torch tensor
        img = FT.to_tensor(img)
        # normalize: already normalized
        img = FT.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        batch_img.append(img.unsqueeze(0))
    batch_img = torch.cat(batch_img, dim=0)
    return [batch_img]

def postprocess_EAST(img, text_boxes_coord):
    """

    :param img: cropped pill image list (PIL image)
    :param text_boxes_coord:
    :return:
    """
    # crop
    img_list = []
    # detected_text_box_num = text_boxes_coord.shape[0]
    # for i in range(detected_text_box_num):
    #     img_list.extend(crop(img[0], text_boxes_coord))
    img_list.extend(crop(img[0], text_boxes_coord))
    return img_list

def preprocess_CRNN(textbox_img_list):
    batch_img = []
    for i in range(len(textbox_img_list)):
        # enhance contrast
        # img = contrast_processing(textbox_img_list[i])
        # resize for SSD
        img = textbox_img_list[i].convert('L')
        img = resize(img, (32, 100))
        # Convert PIL image to Torch tensor
        img = FT.to_tensor(img)
        # normalize: already normalized
        img = FT.normalize(img, [0.5], [0.5])
        batch_img.append(img.unsqueeze(0))
    batch_img = torch.cat(batch_img, dim=0)
    return batch_img

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def preprocess_LSTM(text, rnn_opt):
    source_letter_to_int = rnn_opt['source_letter_to_int']
    target_letter_to_int = rnn_opt['target_letter_to_int']

    # letter2int
    source_batch = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>']) for letter in text]]
    pad_sources_batch = np.array(pad_sentence_batch(source_batch, source_letter_to_int['<PAD>']))
    src_text = pad_sources_batch
    
    pad_targets_batch = np.zeros_like(pad_sources_batch)  ########################################
    trg_int_eos = np.asarray(target_letter_to_int['<EOS>']).reshape(1, -1)  ########################################
    pad_targets_batch = np.append(pad_targets_batch, trg_int_eos, axis=1)
    tmp_trg_text = pad_targets_batch

    return src_text, tmp_trg_text

class PillDataset(data.Dataset):
    def __init__(self, img_dir):
        super(PillDataset, self).__init__()
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.xml_list = glob.glob(self.label_dir + '*.xml')
        self.img_file = img_dir

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, idx):
#         xmllist = glob.glob(self.label_dir + '*.xml')
        
        # load image
#         img_file = re.sub('xml', 'jpg', self.label_file[idx].split('/')[-1], flags=re.IGNORECASE)
#         img_file = self.img_dir.split('/')[-1] #파일이름만
        img_file = self.img_file[idx].split('/')[-1]
        img = Image.open(self.img_file[idx])
#         img = Image.open(self.img_dir)
        
        #print("+++++++++++++++++++++++++++++++++++++++")
        #print(img_file)

        # load label
#         gt_file = xmllist[idx]

        return img, img_file

def make_batch(batch):
    '''

    :param batch: PIL image, list of label text
    :return: Tensor image, list of label text
    :purpose: create list of label text not including tuple
    '''
    labels = []

    for sample in batch:
        img = FT.to_tensor(sample[0])
        filename = sample[1]

    return img, filename
