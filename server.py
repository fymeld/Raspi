import socket
import cv2
import numpy as np
import time
import numpy as np
import readFeature
import process

model_bin = "./model/mask_detector/face_mask_detection.caffemodel"
config_text = "./model/mask_detector/face_mask_detection.prototxt"

h = 480
w = 640

# load tensorflow model
net = cv2.dnn.readNet(model_bin, config_text)

feature,IDS= readFeature.read('face.json')

class VideoServer:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 初始化
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('192.168.137.1', 8002))  # 将套接字绑定到地址
        self.sock.listen(1)  
    
    def Get(self):

        try:
            conn, addr = self.sock.accept()
            print(addr,'已连接...')

            while True:
                time1 =time.time()
                flag = conn.recv(1)
                if flag == b'\xff':

                    head = conn.recv(8)
                    num = int.from_bytes(head,byteorder='big',signed=False)
                    img_data = conn.recv(num)
                    while len(img_data) < num:
                        data = conn.recv(num-len(img_data))
                        if len(data) <= 0:
                            break
                        img_data += data
                    
                    #time2 = time.time()
                    
                    # 将 图片字节码bytes  转换成一维的numpy数组 到缓存中
                    img_buffer_numpy = np.frombuffer(img_data, dtype=np.uint8) 
                    # 从指定的内存缓存中读取一维numpy数据，并把数据转换(解码)成图像矩阵格式
                
                    frame = cv2.imdecode(img_buffer_numpy, 1)

                    image = process.inference(frame)

                    _, img_encode = cv2.imencode('.jpg', image)
                    img_data = img_encode.tobytes()
                    length = len(img_data)
                    head = length.to_bytes(8,byteorder='big',signed=False)

                    fps = 1/(time.time() - time1)
                    cv2.putText(image,"%s:%.2f"%('fps',fps),(10,15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    cv2.imshow('demo', image)
                    k= cv2.waitKey(10) & 0xff
                    if k==27:
                        cv2.destroyWindow('demo')
                        break
                    
                    #time3 = time.time()
    
                    conn.send(b'\xfe')   # 告诉cilent发送下一帧
                    #print('recive time {:.3f},process time {:.3f} send time {:.3f}'.format(time2-time1,time3-time2,time.time()-time3))
                else:
                    head = conn.recv(8)  #读取开头8字节
                    #将读取到的8字节转换成整数，该整数为发送图片的大小
                    num = int.from_bytes(head,byteorder='big',signed=False)
                    #接受图片数据
                    img_data = conn.recv(num)
                    while len(img_data) < num:
                        data = conn.recv(num-len(img_data))
                        if len(data) <= 0:
                            break
                        img_data += data
                    # 将图片字节码bytes转换成一维的numpy数组到缓存中
                    img_buffer_numpy = np.frombuffer(img_data, dtype=np.uint8) 
                    # 从指定的内存缓存中读取一维numpy数据，并把数据转换(解码)成图像矩阵格式
                    frame = cv2.imdecode(img_buffer_numpy, 1)
                    #对接受到的图像进行人脸识别和口罩识别
                    image = process.inference(frame)
                    #将图片编码成jpg格式
                    _, img_encode = cv2.imencode('.jpg', image)
                    img_data = img_encode.tobytes()
                    length = len(img_data)
                    #用8字节表示图片数据大小
                    head = length.to_bytes(8,byteorder='big',signed=False)
                    #发送数据
                    conn.send(head)
                    conn.send(img_data)

            conn.close()
        except:
            conn.close()
        

       

if __name__ == '__main__':
    vs = VideoServer()
    while True:
        vs.Get()
