import socket
import cv2
import numpy as np
import time


class VideoClient:
    def __init__(self,IP,Port,Mode,Vedioindex):
        # 连接服务器（初始化）
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((IP, Port))
        self.mode = Mode
        self.timer = 1/20
        self.Vedioindex = Vedioindex
        
    def Send(self):
        cap = cv2.VideoCapture(self.Vedioindex)
        h = 480
        w = 640
        cap.set(3,w)
        cap.set(4,h)
       
        try:
            k=0
            time1 =time.time()
            while cap.isOpened():
                
                success, frame = cap.read()
                if success:
                    # 将numpy图片转化为bytes字节流
                    _, img_encode = cv2.imencode('.jpg', frame)
                    img_data = img_encode.tobytes()
                    head = len(img_data).to_bytes(8,byteorder='big',signed=False)
                    if self.mode == 0xff:       

                        #print('encode time {:.2f}'.format((time2-time1)*1000))
                        # 连续发送消息
                        self.sock.send(self.mode.to_bytes(1,byteorder='big',signed=False))
                        self.sock.send(head)
                        self.sock.send(img_data)
                        
                        self.sock.recv(1)  #等待下一帧发送信号
                    

                    else:
                        # 连续发送消息
                        self.sock.send(0x00.to_bytes(1,byteorder='big',signed=False))
                        
                        self.sock.send(head)
                        self.sock.send(img_data)
                        #print(len(img_data))
                        
                        head = self.sock.recv(8)
                        num = int.from_bytes(head,byteorder='big',signed=False)
                        img_data = self.sock.recv(num)
                        while len(img_data) < num:
                            data = self.sock.recv(num-len(img_data))
                            #print(len(data))
                            if len(data) <= 0:
                                break
                            img_data += data

                        img_buffer_numpy = np.frombuffer(img_data, dtype=np.uint8)
                        frame = cv2.imdecode(img_buffer_numpy, 1)

                        fps = 1/(time.time() - time1)
                        cv2.putText(frame,"%s:%.2f"%('fps',fps),(10,15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        cv2.imshow('demo', frame)
                        k= cv2.waitKey(10) & 0xff
                        if k==27:
                            break
                        time1 =time.time()
            
            
            cap.release()
            self.sock.close()
        except Exception as e:
            print(e.args)
            cap.release()
            self.sock.close()


if __name__ == '__main__':

    vc = VideoClient('192.168.137.1',8002,0xff,0)
    vc.Send()
    cv2.destroyAllWindows()