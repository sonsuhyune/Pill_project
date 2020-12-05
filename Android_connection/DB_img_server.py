import socket
import time
from PIL import Image
import io
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def send_img(sock, img_path):
    img_f = open(img_path, "rb")
    data = img_f.read()
    data_len = len(data)
    sock.sendall(data_len.to_bytes(4, byteorder="big"))
    print("data length:", data_len)
    
    step = 1024
    loop = int(data_len/step)+1
    
    while len(data) > 0:
        if len(data) < step:
            sock.sendall(data)
            data = []
        else:
            sock.sendall(data[:step])
            data = data[step:]
    
    img_f.flush()
    img_f.close()
    
host = '203.255.176.79'
port = 8089

server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))
server_sock.listen(100)

idx = 0
while True:
    idx += 1
    print("기다리는 중")
    client_sock, addr = server_sock.accept()
    
    img_num = client_sock.recv(4);
    img_num = int.from_bytes(img_num, "little")
    
    print("이미지 개수:", img_num)
    
    for i in range(img_num):
        img_file_name = bytearray(client_sock.recv(1024))[2:]
        img_file_name = img_file_name.decode("utf-8")
        print("이미지 이름:", img_file_name)
        img_path = "img/"+img_file_name
        send_img(client_sock, img_path)
        print("Done")
               
    client_sock.close()

server_sock.close()