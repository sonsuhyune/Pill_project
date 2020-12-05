import socket
import time
import io
import os
import shutil

import torch
from torch.utils.data import DataLoader
import argparse
from PIL import Image
from PIL import ImageFile

from dataset import *
from detect_yolo import detect_pill
from detect_east import detect_text
from detect_crnn import recognize_text

from utils.load_ckpt import load_model
from utils.ssd_eval_utils import *
from yolo_models import *
from east_model import EAST
from crnn_model import CRNN
from rnn_model import Seq2Seq, Encoder, Decoder
from configs import SSD as ssd_opt
from configs import Seq2Seq as rnn_opt
from configs import crnn_opt
from configs import yolo_opt
import matplotlib.pyplot as plt

from nltk.metrics.distance import edit_distance

ImageFile.LOAD_TRUNCATED_IMAGES = True

def write_utf8(s, sock):
    encoded = s.encode(encoding='utf-8')
    sock.sendall(len(encoded).to_bytes(4, byteorder="big"))
    sock.sendall(encoded)


def get_bytes_stream(sock, length):
    buf = b''
    try:
        step = length
        while True:
            data = sock.recv(step)
            buf += data
            if len(buf) == length:
                break
            elif len(buf) < length:
                step = length - len(buf)
    except Exception as e:
        print(e)
    return buf[:length]

'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='for user input image')
    
    parser.add_argument('--data_dir', type=str)

    args = parser.parse_args()
    return args
'''
if __name__ == '__main__':
    #args = parse_args()
    #user_input_img = args.data_dir
    user_input_img =crnn_opt.data_dir
    #print("++++++++++++++++++")
    #print(user_input_img)
    #print("++++++++++++++++++")
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = open(f'result/user_inpt.txt', 'a')
    
    user_input_xml = user_input_img.split('.')[0]
    
    img_dir = ['input/user_input/'+user_input_img]

    
    # load data
    dataset = PillDataset(img_dir)
    pill_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=make_batch)

    # load model
    yolo_net = Darknet(yolo_opt.model_def).to(device)
    # Load checkpoint weights
    yolo_net.load_state_dict(torch.load(yolo_opt.weights_path))
    #print('Finished loading YOLO model!')
    east_net = EAST(False).to(device)
    # east_net = load_model(checkpoint_dir='./weights/east_epoch_200.pth', net=east_net)
    east_net.load_state_dict(torch.load('./weights/east_epoch_245.pth'))
    east_net.eval()
    #print('Finished loading EAST model!')
    # crnn_net = CRNN(crnn_opt).to(device)
    # crnn_net.load_state_dict(torch.load('./weights/crnn_best_accuracy.pth', map_location=device))
    # print('Finished loading CRNN model!')
    input_dim = len(rnn_opt['source_letter_to_int'])
    output_dim = len(rnn_opt['target_letter_to_int'])
    enc = Encoder(input_dim, rnn_opt['enc_dim'], rnn_opt['hid_dim'], rnn_opt['n_layers'], rnn_opt['dropout'])
    dec = Decoder(output_dim, rnn_opt['dec_dim'], rnn_opt['hid_dim'], rnn_opt['n_layers'], rnn_opt['dropout'])
    rnn_net = Seq2Seq(enc, dec, device).to(device)
    rnn_net.load_state_dict(torch.load('weights/rnn-tut1-model.pt'))
    
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    # save_wrong_dir = '/media/yejin/Disk1/wrong/'
    for i, (img, filepath) in enumerate(pill_dataloader):
        ##log.write("="*50+'\n')
        #if filepath.split('_')[3]=='W':
        #    print((filepath.split('_')[3])
        #    continue
        original_image = FT.to_pil_image(img) # remains PIL
        detected_pill_coord = []
        detected_text_coord = []

        # detect pill
        image = preprocess_YOLO(img) #(1,3,416, 416)
        detected_pill_coord = detect_pill(
            yolo_net, original_image, image, iou_thres=yolo_opt.iou_thres, conf_thres=yolo_opt.conf_thres,
            nms_thres=yolo_opt.nms_thres, img_size=yolo_opt.img_size, batch_size=1,)
        if len(detected_pill_coord) == 0:
            ##log.write(filepath+'\n')
            ##log.write("No Pill Found"+'\n')
            print("No pill found")
            # shutil.copy(filepath, save_wrong_dir+'no_pill/')
            length_of_data += 1
            continue
        else:
            cropped_pill_img_list = postprocess_YOLO(original_image, detected_pill_coord) #PIL image
            # plt.imshow(cropped_pill_img_list[0])
            # plt.title("YOLO result")
            # plt.savefig("/media/yejin/Disk1/yolo3_result/"+filepath.split('/')[-1]+'.jpg')
            # plt.show()

        # detect text
        batch_pill_image = preprocess_EAST(cropped_pill_img_list) #[(1, 3, 256, 256)]
        detected_text_coord = detect_text(cropped_pill_img_list, batch_pill_image, east_net, device)
        if len(detected_text_coord) == 0:
            ##log.write(filepath+'\n')
            ##log.write("No textbox Found"+'\n')
            print("No textbox found")
            length_of_data += 1
            # plt.imshow(cropped_pill_img_list[0])
            # plt.title("YOLO result")
            # plt.savefig(save_wrong_dir+filepath.split('/')[-1]+'.jpg')
            continue
        else:
            textbox_list = postprocess_EAST(cropped_pill_img_list, detected_text_coord)

        # recognize text
        batch_text_image = preprocess_CRNN(textbox_list) #(1, 1, 32, 100)
        # plt.imshow(batch_text_image[0][0])
        # plt.title("Preprocess for CRNN")
        # plt.show()
        predicts = recognize_text(batch_text_image)
        #print("pred: ")
        #print(predicts)
        
        # RNN: revise text
        revised_predicts = []
        for _, pred in enumerate(predicts):
            if len(pred) >= 3:
                batch_src_text, tmp_trg_text = preprocess_LSTM(pred, rnn_opt)
                batch_src_text = torch.LongTensor(batch_src_text).transpose(0, 1).to(device)
                tmp_trg_text = torch.LongTensor(tmp_trg_text).transpose(0, 1).to(device)
                output = rnn_net(batch_src_text, tmp_trg_text, 0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                output = output.argmax(1)
                output = "".join([rnn_opt["target_int_to_letter"][i.item()] for i in output if (i!=rnn_opt["target_letter_to_int"]['<EOS>'] or i!=rnn_opt["target_letter_to_int"]['<PAD>'])])
                if output[-5:] == '<EOS>':
                    output = output[:-5]
            else:
                output = pred
            revised_predicts.append(output)
        print(revised_predicts)
        '''
        if(predicts!=gts):
            log.write(filepath+'\n')
            log.write("pred: "+str(predicts)+'\n')
            log.write("gt: "+str(gts)+'\n')
        '''
        
#         log.write(filepath+'\n')
#         log.write("pred: "+str(predicts)+'\n')
#         log.write("gt: "+str(gts)+'\n')
#         #print("=====================")
#         print(predicts)
#         #print("=====================")
#         if len(predicts) > len(labels):
#             for t in range(len(textbox_list)):
#                 plt.imshow(textbox_list[t])
#                 plt.title("EAST result")
#                 #plt.show()
#                 # plt.savefig("/media/yejin/Disk1/connect_EAST/"+filepath.split('/')[-1]+'.jpg')

#         # calculate accuracy and nromed ED
#         length_of_data += len(gts)

#         for p in range(len(predicts)):
#             for g in range(len(gts)):
#                 # accuracy
#                 if predicts[p] in gts[g]: # exactly same
#                     n_correct += 1
#                 # ICDAR2019 Normalized Edit Distance
#                 if len(gts[g]) == 0 or len(predicts[p]) == 0:
#                     norm_ED += 0
#                 elif len(gts[g]) > len(predicts[p]):
#                     norm_ED += 1 - edit_distance(predicts[p], gts[g]) / len(gts[g])
#                 else:
#                     norm_ED += 1 - edit_distance(predicts[p], gts[g]) / len(predicts[p])

    #accuracy = n_correct / float(length_of_data) * 100
    #norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    #print("Accuracy, norm_ED:")
    #print(accuracy, norm_ED)



#     host = '203.255.176.79'
#     port = 8088
#
#     server_sock = socket.socket(socket.AF_INET)
#     server_sock.bind((host, port))
#     server_sock.listen(100)
#
#     idx = 0
#     result_path = './result/'
#     while True:
#         idx += 1
#         print("기다리는 중")
#         client_sock, addr = server_sock.accept()
#
#         len_bytes_string = bytearray(client_sock.recv(1024))[2:]
#         len_bytes = len_bytes_string.decode("utf-8")
#         length = int(len_bytes)
#
#         img_bytes = get_bytes_stream(client_sock, length)
#         # img_path = "pill_img_from_server/img" + str(idx) + str(addr[1]) + ".png"
#         img_path = "pill_img_from_server/35_F_1_W_D_2.jpg"
#         original_image = Image.open(io.BytesIO(img_bytes))
#
#         with open(img_path, "wb") as writer:
#             writer.write(img_bytes)
#         print(img_path + " is saved")
#
#         write_utf8(img_path, client_sock)
#         # model
#         image = preprocess_SSD(original_image)
#         detected_pill_coord = detect_pill(image, min_score=0.2, max_overlap=0.5, top_k=1, visualize=False)
#         image = preprocess_EAST(original_image, detected_pill_coord)
#
#         write_utf8(result_path, client_sock)
#
#         client_sock.close()
#
# server_sock.close()
