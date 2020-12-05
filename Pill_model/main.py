import socket
import time
import io
import os

import torch
from torch.utils.data import DataLoader

from PIL import Image
from PIL import ImageFile

from dataset import *
from detect_ssd import detect_pill
from detect_east import detect_text
from detect_crnn import recognize_text

from utils.load_ckpt import load_model
from utils.ssd_eval_utils import *
from model import build_ssd
from east_model import EAST
from crnn_model import CRNN
from configs import SSD as ssd_opt
from configs import opt as crnn_opt
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = 'input/'
    gt_dir = 'input/'

    # load data
    dataset = PillDataset(img_dir, gt_dir)
    pill_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=make_batch)

    # load model
    ssd_net = build_ssd(opt, size=300, num_classes=opt['num_classes'] + 1)
    ssd_net = load_model(checkpoint_dir='./weights/ssd_epoch_0400_loss_1.9637260580.pth', net=ssd_net)
    ssd_net = ssd_net.to(device)
    ssd_net.eval()
    print('Finished loading SSD model!')
    east_net = EAST(False).to(device)
    # east_net = load_model(checkpoint_dir='./weights/east_epoch_200.pth', net=east_net)
    east_net.load_state_dict(torch.load('./weights/east_epoch_200.pth'))
    east_net.eval()
    print('Finished loading EAST model!')
    # crnn_net = CRNN(crnn_opt).to(device)
    # crnn_net.load_state_dict(torch.load('./weights/crnn_best_accuracy.pth', map_location=device))
    # print('Finished loading CRNN model!')

    for i, (img, labels) in enumerate(pill_dataloader):
        #print("여기이이이이")
        original_image = FT.to_pil_image(img) # remains PIL
        detected_pill_coord = []
        detected_text_coord = []

        # detect pill
        image = preprocess_SSD(img) #(1,3,300,300)
        detected_pill_coord = detect_pill(original_image, image, ssd_net, min_score=0.2, max_overlap=0.5, top_k=1, visualize=False)
        if len(detected_pill_coord) == 0:
            print("No pill found")
            continue
        else:
            cropped_pill_img_list = postprocess_SSD(original_image, detected_pill_coord) #PIL image
            plt.imshow(cropped_pill_img_list[0])
            plt.title("SSD result")
            plt.savefig("SSD_result.jpg")
            #plt.show()

        # detect text
        batch_pill_image = preprocess_EAST(cropped_pill_img_list) #[(1, 3, 256, 256)]
        detected_text_coord = detect_text(cropped_pill_img_list, batch_pill_image, east_net, device)
        if len(detected_text_coord) == 0:
            print("No textbox found")
            continue
        else:
            textbox_list = postprocess_EAST(cropped_pill_img_list, detected_text_coord)
            
            plt.imshow(textbox_list[0])
            plt.title("EAST result")
            plt.savefig("EAST_result.jpg")
            #plt.show()

        # recognize text
        batch_text_image = preprocess_CRNN(textbox_list) #(1, 1, 32, 100)
        plt.imshow(batch_text_image[0][0])
        plt.title("Preprocess for CRNN")
        plt.savefig('CRNN_prepro_Result.jpg')
        #plt.show()
        recognized_text = recognize_text(batch_text_image, labels)
        print(recognize_text)



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
