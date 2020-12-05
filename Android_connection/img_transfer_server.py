import socket
import time
from PIL import Image
import io
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def write_utf8(s, sock):
    encoded = s.encode(encoding='utf-8')
    sock.sendall(len(encoded).to_bytes(4, byteorder="big"))
    sock.sendall(encoded)

host = '203.255.176.79'
port = 8088

server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))
server_sock.listen(100)

idx = 0
while True:
    idx += 1
    print("기다리는 중")
    client_sock, addr = server_sock.accept()
    
    len_bytes_string = bytearray(client_sock.recv(1024))[2:]
    len_bytes = len_bytes_string.decode("utf-8")
    length = int(len_bytes)

    img_bytes = get_bytes_stream(client_sock, length)
    img_path = "pill_img_from_server/img"+str(idx)+str(addr[1])+".png"
   
    with open(img_path, "wb") as writer:
        writer.write(img_bytes)
    print(img_path+" is saved")

    write_utf8(img_path+","+"result", client_sock)
    
    client_sock.close()

server_sock.close()