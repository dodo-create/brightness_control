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

cap = cv2.VideoCapture(0)

while True: 
    # Reads the video 
    _,frame = cap.read()

    #FLips images
    frame = cv2.flip(frame,1)

    #convert to rgb
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    #process the image
    Process = hands.process(frameRGB)

    landmarkList = []

    #to check if hands are in frame
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id,landmarks in enumerate(handlm.landmark):

                height,width,color = frame.shape


                x,y = int(landmarks.x*width), int(landmarks.y*height)
                landmarkList.append([_id,x,y])
            
            Draw.draw_landmarks(frame,handlm,mpHands.HAND_CONNECTIONS)

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
