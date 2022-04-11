# import necessary packages
from __future__ import print_function
import numpy as np
import cv2
import os

#Initialize 'name' to trigger only when a new person is identified.
#create a data base or folder on the name initialised in specified location.
name=input('Enter employee name: ')
path=  "C:\\Users\\Mypc\\Desktop\\temp\\"
os.chdir(path)
os.makedirs(name)

#Webcam initialized with LED indication.
cam=cv2.VideoCapture(0)

#Capture Window is customised with width and hight as per requirement.
cv2.namedWindow('output',cv2.WINDOW_NORMAL)
cv2.resizeWindow('output',500,300)

#ImageCount statrs with initial count 'zero'
img_counter=0

#Check the  condition for true and initialize capture process.

while True:
    ret, frame = cam.read()

    if not ret:
        print("failed to grab frame")
        break
   
    #An transparent or overlay frame is created, to display instruction while capturing image.
    overlay=frame.copy()
    output=frame.copy()
    cv2.rectangle(overlay, (200,150), (400,385),(0,255,0), -1)
    
    #Starts the image capture process with instructions on the capture frame step by step.
    if img_counter==0:
        cv2.putText(overlay,"Look Straight",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    elif img_counter==1:
        cv2.putText(overlay,"Turn Face Left",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    elif img_counter==2:
        cv2.putText(overlay,"Turn Face Right",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    elif img_counter==3:
        cv2.putText(overlay,"Turn Face 45 Deg right",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    elif img_counter==4:
        cv2.putText(overlay,"Turn Face 45 Deg Left",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    elif img_counter==5:
        cv2.putText(overlay,"Move a step back",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    elif img_counter==6:
        cv2.putText(overlay,"Move a step forward",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    elif img_counter==7:
        cv2.putText(overlay,"capture face smiling",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    elif img_counter==8:
        cv2.putText(overlay,"capture with mask",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    elif img_counter==9:
        cv2.putText(overlay,"Turn Left with mask",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
    else:
        cv2.putText(overlay,"Turn Right with mask",(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        alpha=0.2
        break
            
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