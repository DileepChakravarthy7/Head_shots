#import necessary libraries 
import numpy as np
import mediapipe as mp
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import os

#Initialize 'name' to trigger only when a new person is identified.
#create a data base or folder in the employee name or ID is  initialised in specified location.
#employee name or ID, path to store image is managed through backend to frontend api
name=input('Enter employee name: ')
path=  "C:\\Users\\Mypc\\Desktop\\temp\\"
os.chdir(path)
os.makedirs(name)

#Webcam initialized with LED indication.
cam=cv2.VideoCapture(0)
cv2.namedWindow('output',cv2.WINDOW_NORMAL)
cv2.resizeWindow('output',500,300)

detector = FaceMeshDetector(maxFaces=1)
mp_drawings=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic


#ImageCount statrs with initial count 'zero'
img_counter=0
case=img_counter
degree_sign = u"\N{DEGREE SIGN}"
def switchcase(case):
    switch={0:"Look Straight",
            1:"Turn Face Left",
            2:"Turn Face Right",
            3:"Turn 45"+degree_sign+"right",
            4:"Turn 45"+degree_sign+"left",
            5:"Move a step back",
            6:"Move a step forward",
            7:"capture face smiling",
            8:"capture with mask",
            9:"Turn Left with mask",
            }
    return switch.get(case)
    

#define percentage of detection confidence through mediapipe required.
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cam.read()
        roi = frame[180:375,200:400]
       
        if not ret:
            print("failed to grab frame")
            break
        
        results = holistic.process(frame)
       #frame, faces = detector.findFaceMesh(frame)   
        overlay=frame.copy()
        output=frame.copy()
        cv2.rectangle(overlay, (200,180), (400,375),(255,0,0), 4)
        
        if results.face_landmarks:
            mp_drawings.draw_landmarks(overlay,results.face_landmarks,mp_holistic.FACE_CONNECTIONS,
                                       mp_drawings.DrawingSpec(color=(110,150,10),thickness=2,circle_radius=2),
                                       mp_drawings.DrawingSpec(color=(256,80,121),thickness=2,circle_radius=2)
                                           )
            mp_drawings.draw_landmarks(overlay,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                                       mp_drawings.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                                       mp_drawings.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                           )
        else:
            print("Find Face")
        cv2.putText(overlay,switchcase(img_counter),(130,450),cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),3)
        alpha=0.2
        cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)  
        cv2.imshow('output',output)
              
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name ="C:\\Users\\Mypc\\Desktop\\temp\\"+ name +"/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name,roi)
            print("{} written!".format(img_name))
            img_counter += 1
            if img_counter==10:
                break
cam.release()
cv2.destroyAllWindows()
