import cv2
import opencv_dnn_infer
import readFeature
import cosine_distance
import time

model_bin = "./model/mask_detector/face_mask_detection.caffemodel"
config_text = "./model/mask_detector/face_mask_detection.prototxt"

id2class = {0: 'Mask', 1: 'NoMask'}
id2chiclass = {0: '您戴了口罩', 1: '您没有戴口罩'}
colors = ((0, 255, 0), (255, 255 , 0))

# load tensorflow model
net = cv2.dnn.readNet(model_bin, config_text)
net2 = cv2.dnn.readNet('./model/face_recognition_sface_2021dec.onnx')

cap = cv2.VideoCapture(1)
h = 480
w = 640
cap.set(3,w)
cap.set(4,h)

feature,IDS= readFeature.read('face.json')

while True:
    time1 = time.time()
    ret,img = cap.read()
    image = cv2.flip(img,1)

    # 人脸检测 mask detection
    time1 = time.time()
    detect_results=opencv_dnn_infer.inference(net,image[:,:,::-1],target_shape=(260,260))

    for position,class_id,confidence in detect_results:
        #(top,right,bottom,left)
        top = min(int(position[0]*h),h)
        right = min(int(position[1]*w),w)
        bottom = max(0,int(position[2]*h))
        left = max(0,int(position[3]*w))

        face = image[bottom:top,left:right]

        #SFace model input shape(3,112,112)
        blobImage = cv2.dnn.blobFromImage(face, 1.0, (112, 112))
        net2.setInput(blobImage)
        img_feature = net2.forward()
        for l,img_encode in enumerate(img_feature):

            compare_results = cosine_distance.face_distance(feature,img_encode)

            for i,res in enumerate(compare_results) :
                if res > 0.6:
                    cv2.putText(image,"%s:%.2f"%(IDS[i],res),(left+4,top+13),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            cv2.putText(image, "%s" % (id2class[class_id]), (left + 2, bottom - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id])        
            cv2.rectangle(image, (left,bottom), (right,top), colors[class_id], thickness=2)
    
    fps = 1/(time.time() - time1)
    cv2.putText(image,"%s:%.2f"%('fps',fps),(10,15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imshow('demo', image)
    k= cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()