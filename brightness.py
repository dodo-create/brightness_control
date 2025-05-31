#Libraries needed
import cv2 #for camera control
import mediapipe as mp  #for media processing 
from math import hypot #hypotenous 
import screen_brightness_control as sbc #controlling brighntness
import numpy as np #array processing

#=================Initializing the model=======================
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode = False, #specifies if input is static image or stream 
    model_complexity = 1, #accuracy
    min_detection_confidence = 0.75, #minimum confidence value for success
    min_tracking_confidence = 0.75, #minimum confidence value for tracking
    max_num_hands = 2 #Maximum number of hands
)

Draw = mp.solutions.drawing_utils

#starts the camera 

cap = cv2.VideoCapture(0)

#reads video frame by frame 
_,frame = cap.read()
    
#Flip image 
frame=cv2.flip(frame,1)
while True: 
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break