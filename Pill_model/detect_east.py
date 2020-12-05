import torch
from torchvision import transforms
from torch.utils import data
from PIL import Image, ImageDraw
from east_model import EAST
import os
import glob
import numpy as np
import lanms
import math
#import cv2

def get_rotate_mat(theta):
	'''positive theta value means rotate clockwise'''
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	with torch.no_grad():
		score, geo = model(img.to(device))
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	# return adjust_ratio(boxes, ratio_w, ratio_h)
	return boxes

def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	return img


def detect_text(original_image, process_image, model, device):
	"""

	:param original_image: cropped pill image list that contains real size [ PIL image ]
	:param process_image: batch image resized for EAST [[1,3,256,256]]
	:param model: EAST
	:param device: gpu
	:return:
	"""

	for i, pill_image in enumerate(process_image):
		boxes = detect(pill_image, model, device)
		if boxes is None:
			boxes = []
			return boxes

		else:
			for j in range(len(boxes)):
				ratio_w = process_image[i].shape[-1] / original_image[i].width
				ratio_h = process_image[i].shape[-2] / original_image[i].height
				boxes[j, [0, 2, 4, 6]] /= ratio_w
				boxes[j, [1, 3, 5, 7]] /= ratio_h

	# get top-1 box
	# ind = np.argmax(boxes[:, -1])
	# boxes = boxes[ind]

	# seq = []
	# if boxes is not None:
	# 	seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
	# with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
	# 	f.writelines(seq)
	# os.chmod(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 0o777)
	# return boxes[np.newaxis, :-1]
	return boxes[:, :-1]

if __name__ == '__main__':
	# img_path    = '../ICDAR_2015/test_img/img_2.jpg'
	# model_path  = './pths/east_vgg16.pth'
	# res_img     = './res.bmp'
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = "1"

	model_path = './pths_rotate/model_epoch_245.pth'
	res_dir = '/media/yejin/Disk1/east_result_rotate_245_2/'

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST().to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()

	# import pandas as pd
	# excel_file = '/home/yejin/Git/EAST/east_dataset.xlsx'
	# data_idx = 2
	# df_idx = pd.read_excel(excel_file, sheet_name='train_val_test_idx')
	# mode_idx = df_idx.index[df_idx[0] == data_idx].to_list()
	# df_filename = pd.read_excel(excel_file, sheet_name='filenames')
	# filepath = df_filename.loc[mode_idx, :].values.tolist()

	# for i, im in enumerate(filepath):
	# 	print("Test // %s" %im)
	# 	img = Image.open(im[0]).convert('RGB')
	# 	boxes = detect(img, model, device)
	# 	plot_img = plot_boxes(img, boxes)
	# 	plot_img.save(res_dir+im[0].split('/')[-1])

	img_path = '/media/yejin/Disk1/Text_Detection_Data/rename_img/'
	imgs = glob.glob(img_path + '*')
	if not os.path.exists(res_dir):
		os.mkdir(res_dir)
	for i, im in enumerate(imgs):
		img = Image.open(im).convert('RGB')
		boxes = detect(img, model, device)
		plot_img = plot_boxes(img, boxes)
		plot_img.save(res_dir + im.split('/')[-1])


