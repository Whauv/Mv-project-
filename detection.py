import cv2 as cv
import numpy as np

thres = 0.5

capture = cv.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

classNames = []
classFile = 'coco.names'
with open(classFile,'r') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

model = cv.CascadeClassifier("face_detector.xml") 
configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'

net = cv.dnn_DetectionModel(weightspath, configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    boolean, frame = capture.read()
    classIds, confs, bbox = net.detect(frame,confThreshold = thres)
    print(classIds,bbox)

    if boolean == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        coordinate_list = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) 

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
                cv.rectangle(frame,box,color = (0, 255,0),thickness = 2)
                cv.putText(frame,classNames[classId - 1].upper(),(box[0] + 10,box[1]+ 30),
                cv.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),2)
                cv.putText(frame,str(round(confidence*100,2)),(box[0] + 200,box[1]+ 30),
                cv.FONT_HERSHEY_COMPLEX,1,(0, 255, 0),2) 
                
            for (x,y,w,h) in coordinate_list:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
                
            cv.imshow("Face Detection", frame)
            
            if cv.waitKey(20) == ord('a'):
                break
        
capture.release()
cv.destroyAllWindows()