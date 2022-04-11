# import necessary packages
from __future__ import print_function
import numpy as np
import mediapipe as mp
import cv2
import os

#Initialize 'name' to trigger only when a new person is identified.
#create a data base or folder in the employee name or ID is  initialised in specified location.
#employee name or ID, path to store image is managed through backend to frontend api
name=input('Enter employee name: ')
path=  "C:\\Users\\Mypc\\Desktop\\temp\\"
os.chdir(path)
os.makedirs(name)

#define the parameters of mediapipe solution.
mp_drawings=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic

#Webcam initialized with LED indication.
cam=cv2.VideoCapture(0)
cv2.namedWindow('output',cv2.WINDOW_NORMAL)
cv2.resizeWindow('output',500,300)

#ImageCount statrs with initial count 'zero'
img_counter=0

#define percentage of detection confidence through mediapipe required.

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    #Check the  condition for true and initialize capture process.
    while True:
        ret, frame = cam.read()
    
        if not ret:
            print("failed to grab frame")
            break
        
        #image in the frame is processed with mediapipe function.
        results = holistic.process(frame)
        
        #An transparent or overlay frame is created, to display instruction while capturing image.
        overlay=frame.copy()
        output=frame.copy()
        cv2.rectangle(overlay, (200,150), (375,350),(0,255,0), -1)
        
        #initialiing the landmarks on face when candidate is infront of camera
        mp_drawings.draw_landmarks(overlay,results.face_landmarks,mp_holistic.FACE_CONNECTIONS,
                                    mp_drawings.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2),
                                    mp_drawings.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2)
                                       )
        mp_drawings.draw_landmarks(overlay,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                                   mp_drawings.DrawingSpec(color=(245,0,0),thickness=2,circle_radius=4),
                                   mp_drawings.DrawingSpec(color=(245,0,0),thickness=2,circle_radius=2)
                                   )
        
        #Starts the image capture process with instructions on the capture frame step by step.
        if img_counter==0:
            cv2.putText(overlay,"Look Straight",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        elif img_counter==1:
            cv2.putText(overlay,"Turn Face Left",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        elif img_counter==2:
            cv2.putText(overlay,"Turn Face Right",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        elif img_counter==3:
            cv2.putText(overlay,"Turn Face 45deg right",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        elif img_counter==4:
            cv2.putText(overlay,"Turn Face 45deg Left",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        elif img_counter==5:
            cv2.putText(overlay,"Move a step back",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        elif img_counter==6:
            cv2.putText(overlay,"Move a step forward",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        elif img_counter==7:
            cv2.putText(overlay,"capture face smiling",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        elif img_counter==8:
            cv2.putText(overlay,"capture with mask",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        elif img_counter==9:
            cv2.putText(overlay,"Turn Left with mask",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
        else:
            cv2.putText(overlay,"Turn Right with mask",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            alpha=0.2
            break
        
        #image count is terminated after break and goes back to image count zero again when reinitialised the capture process.
        #After each count image is captured and output image is saved in specified location. 
        cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)    
        cv2.imshow('output',output)
                 
        
        k = cv2.waitKey(1)
        if k%256 == 27: # ESC pressed
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "C:\\Users\\Mypc\\Desktop\\temp\\"+name+"/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    
cam.release()
    
cv2.destroyAllWindows()