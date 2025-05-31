# Libraries needed
import cv2  # For camera control
import mediapipe as mp  # For hand tracking
from math import hypot  # Hypotenuse calculation
import screen_brightness_control as sbc  # Controlling brightness
import numpy as np  # Array processing

# =================Initializing the model=======================
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,  # Specifies if input is static image or stream
    model_complexity=1,  # Accuracy
    min_detection_confidence=0.75,  # Minimum confidence value for success
    min_tracking_confidence=0.75,  # Minimum confidence value for tracking
    max_num_hands=2  # Maximum number of hands
)

Draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def get_environment_brightness(frame):
    """Estimate the ambient brightness of the environment using histogram analysis."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    brightness = np.mean(gray)  # Compute the average brightness
    return np.interp(brightness, [50, 200], [20, 100])  # Map it to screen brightness range

while True:
    # Reads the video
    _, frame = cap.read()

    # Flip the image
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image
    Process = hands.process(frameRGB)

    landmarkList = []

    # Auto-adjust brightness based on environment
    env_brightness = get_environment_brightness(frame)
    sbc.set_brightness(int(env_brightness))

    # Check if hands are detected
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, _ = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

        if landmarkList:
            x1, y1 = landmarkList[4][1], landmarkList[4][2]  # Thumb coordinates
            x2, y2 = landmarkList[8][1], landmarkList[8][2]  # Index coordinates

            cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 7, (0,255,0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draws line between fingers

            L = hypot(x2 - x1, y2 - y1)  # Calculates the distance

            b_level = np.interp(L, [15, 220], [0, 100])  # Map hand range to brightness range
            sbc.set_brightness(int(b_level))  # Adjust brightness

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()