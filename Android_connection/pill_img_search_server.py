import socket
import time
from PIL import Image
import io
from PIL import ImageFile
import subprocess
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

def write_utf8(s, sock):
    encoded = s.encode(encoding='utf-8')
    sock.sendall(len(encoded).to_bytes(4, byteorder="big"))
    sock.sendall(encoded)

def get_bytes_stream(sock, length):
    buf = b''
    try:
        step = length
        #print("length: ", length)
        while True:
            #print("step:", step)
            data = sock.recv(step)
            buf += data
            if len(buf) == length:
                break
            elif len(buf) < length:
                step = length - len(buf)
    except Exception as e:
        print(e)
    return buf[:length]

host = '203.255.176.79'
port = 8088

server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))
server_sock.listen(100)
result = ''
idx = 0
while True:
    idx += 1
    print("기다리는 중")
    client_sock, addr = server_sock.accept()

    len_bytes_string = bytearray(client_sock.recv(1024))[2:]
    len_bytes = len_bytes_string.decode("utf-8")
    length = int(len_bytes)
    
    recieve_start = time.time()
    img_bytes = get_bytes_stream(client_sock, length)
    img_path = "pill_img_from_server/img"+str(idx)+str(addr[1])+".jpg"
    
    with open(img_path, "wb") as writer:
        writer.write(img_bytes)
    print(img_path+" is saved")
    recieve_end = time.time()
    print("recieve time: ", str(recieve_end-recieve_start))
    
    model_start = time.time()
    img_name = 'img'+str(idx)+str(addr[1])+".jpg"
    # 사용자이미지(img"+str(idx)+str(addr[1])+".png")를 ../Pill_model/input/user_input 밑으로 이동
    #move_result = subprocess.check_output(["cp", "pill_img_from_server/"+img_name,"../Pill_model/input/user_input/"+img_name])
    move_result = subprocess.check_output(["cp", "pill_img_from_server/"+img_name,"../Pill_model/input/user_input/"+img_name])
    
    #모델 실행
    temp_result = subprocess.check_output(["python", "main_YEC.py","--data_dir",img_name], cwd="../Pill_model")
    
    ##############shape##################     
    files = ['../img/cnn_shape/test/circle/'+i for i in os.listdir('../img/cnn_shape/test/circle/')] 
    subprocess.call(['rm','-r'] +files)   
    
    shape_move_result = subprocess.check_output(["cp", "pill_img_from_server/"+img_name,"../img/cnn_shape/test/circle/"+img_name])
    
    
    shape_temp_result = subprocess.check_output(["python", "test.py"], cwd="../pytorch-image-classification")  
    model_end = time.time()
    print("model time: ", str(model_end-model_start))
    
    time.sleep(20)
    
    shape_result = str(shape_temp_result)
    shape_result = shape_result[2:-3]   
    temp_result=str(temp_result)
    result = temp_result[3:-4]
    print(temp_result)
    print("MARK: ",result)
    print("SHAPE: ",str(shape_result))
    
    write_utf8(img_path, client_sock)
    write_utf8(result, client_sock)
    write_utf8(shape_result, client_sock)
    
    client_sock.close()

server_sock.close()