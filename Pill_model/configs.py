import argparse
import string

SSD = {
    'num_classes': 1,
    'labelmap': {'background': 0, 'pill': 1},
    # VGG
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300, #or 300
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3, 4], [2, 3, 4], [2], [2]],
    # 'variance': [0.1, 0.2]
    'variance': [1, 1],
    'clip': True,
    'name': 'PILL'
}

# CRNN
crnn_parser = argparse.ArgumentParser(description='CRNN')
# parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
crnn_parser.add_argument('--benchmark_all_eval', default=False, help='evaluate 10 benchmark evaluation datasets')
crnn_parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
crnn_parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
# parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
""" Data processing """
crnn_parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
crnn_parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
crnn_parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
crnn_parser.add_argument('--rgb', action='store_true', help='use rgb input')
crnn_parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
crnn_parser.add_argument('--sensitive', default=True, help='for sensitive character mode')
crnn_parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
crnn_parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
""" Model Architecture """
crnn_parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
crnn_parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
crnn_parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
crnn_parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
crnn_parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
crnn_parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
crnn_parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
crnn_parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
crnn_parser.add_argument('--data_dir', type=str)
crnn_opt = crnn_parser.parse_args()
if crnn_opt.sensitive:
        crnn_opt.character = string.printable[:-6]

# YOLOv3
yolo_parser = argparse.ArgumentParser(description='YOLOv3')
yolo_parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
yolo_parser.add_argument("--model_def", type=str, default="yolo_utils/yolov3-custom.cfg", help="path to model definition file")
yolo_parser.add_argument("--data_config", type=str, default="config/custom_test.data", help="path to data config file")
yolo_parser.add_argument("--weights_path", type=str, default="weights/yolov3_ckpt_499.pth", help="path to weights file")
yolo_parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
yolo_parser.add_argument("--iou_thres", type=float, default=0.1, help="iou threshold required to qualify as detected")
yolo_parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
yolo_parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
yolo_parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
yolo_parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
yolo_parser.add_argument('--data_dir', type=str)
yolo_opt = yolo_parser.parse_args()

# Seq2Seq
Seq2Seq = {
    'source_letter_to_int': {'<PAD>': 0, '<UNK>': 1, '<GO>': 2, '<EOS>': 3, 'O': 4, '.': 5, 'Z': 6, '2': 7, 'K': 8, 'X': 9, '5': 10, 'o': 11, 'D': 12, 'j': 13, 'A': 14, 'e': 15, \
                            'T': 16, 'y': 17, 's': 18, ')': 19, 'x': 20, 'c': 21, 'E': 22, '(': 23, 'W': 24, 'I': 25, 'F': 26, '/': 27, 'r': 28, 'B': 29, '>': 30, '4': 31, 'C': 32, \
                            'm': 33, 'n': 34, 'R': 35, '0': 36, 'V': 37, '-': 38, 'w': 39, 'L': 40, 'Y': 41, 'z': 42, '8': 43, 'P': 44, 'S': 45, '6': 46, 'U': 47, '3': 48, 't': 49, \
                            'N': 50, '7': 51, ':': 52, 'i': 53, 'a': 54, 'l': 55, 'H': 56, '1': 57, '|': 58, '9': 59, 'M': 60, 'G': 61, 'J': 62},

    'target_letter_to_int': {'<PAD>': 0, '<UNK>': 1, '<GO>': 2, '<EOS>': 3, 'j': 4, 'N': 5, 'Z': 6, 'V': 7, '4': 8, '2': 9, 'w': 10, 'A': 11, 'z': 12, 'n': 13, '6': 14, 'i': 15, \
                           '/': 16, '.': 17, 'e': 18, 'T': 19, 'L': 20, '8': 21, 'r': 22, 'o': 23, 'W': 24, '|': 25, 'c': 26, 's': 27, '7': 28, '1': 29, 'F': 30, 'H': 31, '9': 32, \
                           'S': 33, 'D': 34, 'y': 35, '3': 36, 'P': 37, 'x': 38, '0': 39, 'G': 40, '5': 41, 'X': 42, '-': 43, 'J': 44, 't': 45, 'C': 46, 'B': 47, 'm': 48, 'l': 49, \
                           'a': 50, 'M': 51, 'R': 52, 'U': 53, 'K': 54, 'I': 55, 'E': 56, 'O': 57, 'Y': 58},
    
    'target_int_to_letter': {0: '<PAD>', 1: '<UNK>', 2: '<GO>', 3: '<EOS>', 4: 'j', 5: 'N', 6: 'Z', 7: 'V', 8: '4', 9: '2', 10: 'w', 11: 'A', 12: 'z', 13: 'n', 14: '6', 15: 'i', 16: '/', 17: '.', 18: 'e', 19: 'T', 20: 'L', 21: '8', 22: 'r', 23: 'o', 24: 'W', 25: '|', 26: 'c', 27: 's', 28: '7', 29: '1', 30: 'F', 31: 'H', 32: '9', 33: 'S', 34: 'D', 35: 'y', 36: '3', 37: 'P', 38: 'x', 39: '0', 40: 'G', 41: '5', 42: 'X', 43: '-', 44: 'J', 45: 't', 46: 'C', 47: 'B', 48: 'm', 49: 'l', 50: 'a', 51: 'M', 52: 'R', 53: 'U', 54: 'K', 55: 'I', 56: 'E', 57: 'O', 58: 'Y'},


    'enc_dim': 15,
    'dec_dim': 15,
    'hid_dim': 50,
    'n_layers': 2,
    'dropout': 0
}
