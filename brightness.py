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

        if landmarkList != []:
            x1 , y1 = landmarkList[4][1], landmarkList[4][2] #xy coordinates of thumb

            x2 , y2 = landmarkList[8][1], landmarkList[8][2] # xy coordinates of index

            cv2.circle(frame,(x1,y1),7,(0,255,0),cv2.FILLED)
            cv2.circle(frame,(x2,y2),7,(0,255,0),cv2.FILLED)

            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)  #draws a line from thumb to index

            L = hypot(x2 - x1, y2 - y1) #calculates the hypotenous

            b_level = np.interp(L,[15,220],[0,100]) #hand range 15-220 brightness ranges from 0-100

            sbc.set_brightness(int(b_level)) #adjusts brightness

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
