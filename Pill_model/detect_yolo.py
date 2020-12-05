from __future__ import division

from yolo_models import *
from yolo_utils.utils import *
from yolo_utils.datasets import *
from yolo_utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_pill(model, original_img, imgs, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # labels = []
    # sample_metrics = []  # List of tuples (TP, confs, pred)

    imgs = Variable(imgs.type(Tensor), requires_grad=False)

    with torch.no_grad():
        outputs = model(imgs.to(device))
        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        if outputs[0] == None:
            outputs = []
            return outputs
        # sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    # true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    # precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # get top-1 box
    _, top_index = torch.max(outputs[0][:, 4], 0)
    outputs = outputs[0][top_index].unsqueeze(0)

    # Rescale boxes to original image
    outputs = rescale_boxes(outputs, imgs.shape[-1], np.array(original_img).shape[:2])

    # check RESULT
    # fig, ax = plt.subplots(1)
    # # image = imgs[0].transpose(0,2).detach().cpu().numpy()
    # ax.imshow(original_img)
    # for i in range(len(outputs)):
    #     xmin = outputs[i, 0]
    #     ymin = outputs[i, 1]
    #     xmax = outputs[i, 2]
    #     ymax = outputs[i, 3]
    #     width = xmax - xmin
    #     height = ymax - ymin
    #     rect = patches.Rectangle((xmin, ymin), width, height, edgecolor="blue", fill=False)
    #     ax.add_patch(rect)
    # plt.show()

    return outputs[:, :4]