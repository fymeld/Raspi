import face_recognition as recognizer
import cv2
import os
import numpy
import json
import opencv_dnn_infer
import cosine_distance

model_bin = "./model/mask_detector/face_mask_detection.caffemodel"
config_text = "./model/mask_detector/face_mask_detection.prototxt"
h = 480
w = 640

# load tensorflow model
net = cv2.dnn.readNet(model_bin, config_text)
net2 = cv2.dnn.readNet('./model/face_recognition_sface_2021dec.onnx')

cap = cv2.VideoCapture(1)

cap.set(3,w)
cap.set(4,h)

name = input('please input your ID\n')

datadir=os.path.join('dataset',name)

if not os.path.exists(datadir):
    os.makedirs(datadir)

i=0
_,_ = cap.read()

batch_features=[]

while i<50:
    ret,img = cap.read()
    image = cv2.flip(img,1)
    
    #detect
    detect_results=opencv_dnn_infer.inference(net,image[:,:,::-1])
    
    # 绘制检测矩形
    for position,_,_ in detect_results:
        top = min(int(position[0]*h),h)
        right = min(int(position[1]*w),w)
        bottom = max(0,int(position[2]*h))
        left = max(0,int(position[3]*w))
            
        face = image[bottom:top,left:right]
        landmarks = recognizer.face_landmarks(face[:,:,::-1],model='small')

        #SFace model input shape(3,112,112)
        blobImage = cv2.dnn.blobFromImage(face, 1.0, (112, 112))
        net2.setInput(blobImage)
        img_feature = net2.forward()
        for l,img_encode in enumerate(img_feature):
            batch_features.append(img_encode)
            i+=1

        cv2.rectangle(image, (left,bottom), (right,top), (255, 0, 0), thickness=2)
        cv2.putText(image, name, (left,bottom-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('demo', image)
    k= cv2.waitKey(30) & 0xff

    if k==27:
        break

features= numpy.array(batch_features)
features = features.reshape((-1,128))
feature = numpy.mean(features,0)


print(recognizer.face_distance(feature,features))

try:
    with open('face.json','r') as f:

        featuredict=json.load(f)
        f.close()
except:
    featuredict={}

featuredict = dict(featuredict)
w_dict = {name:[i.item() for i in feature]}
for key in featuredict:
    score = cosine_distance.face_distance([numpy.array(featuredict[key])],feature)
    if score > 0.6:
       w_dict[key]=featuredict[key] 
        
        
with open('face.json','w+') as f:
    f.write(json.dumps(w_dict))
    f.close()

cap.release()
cv2.destroyAllWindows()