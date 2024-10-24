import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime

# Constants
WIDTH, HEIGHT = 640, 480  # Screen size
MALLET_RADIUS = 30  # Radius of the player's mallet
PUCK_RADIUS = 20  # Radius of the puck
SPEED_LIMIT = 15  # Limit puck speed
GOAL_SIZE = 120  # Width of the goal

# Colors
WHITE = (255, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
def hand_detector(frame):
    # Initialize MediaPipe Hands
      mp_hands = mp.solutions.hands
      hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Detector', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Physics variables for puck
puck_pos = np.array([640 // 2, 480 // 2], dtype=np.float32)
puck_vel = np.array([3, 3], dtype=np.float32)  # Starting velocity
# Function to detect mallet using color tracking (red object)
def detect_mallet(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours for the mallet
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            return int(x), int(y), int(radius)
    return None

# Function to update puck physics (movement and collision)
def update_puck():
    global puck_pos, puck_vel

    # Update puck position
    puck_pos += puck_vel
  # Collision with walls
    if puck_pos[0] - 20 <= 0 or puck_pos[0] + 20 >= 640:
        puck_vel[0] *= -1  # Bounce horizontally

    if puck_pos[1] - 20 <= 0 or puck_pos[1] + 20 >= 480:
        puck_vel[1] *= -1  # Bounce vertically

    # Clamp speed
    speed = np.linalg.norm(puck_vel)
    if speed > 15:
        puck_vel = (puck_vel / speed) * 15

# Function to check for collision with the mallet
def check_collision(mallet_pos):
    global puck_pos, puck_vel

    dist = np.linalg.norm(puck_pos - mallet_pos)
    if dist <= 30 + 20:
        # Calculate the collision direction
        direction = (puck_pos - mallet_pos) / dist
        puck_vel = direction * 15  # Reflect puck with new velocity  
#def scoreboard(score1,score2):


# Main loop for the game
duration=5
cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
quality=0
while True:
    ret, frame = cap.read()
    start_time= datetime.now()
    difference=(datetime.now()-start_time).seconds
    frame = cv2.resize(frame, (400, 400))
# Detect the mallet position
    mallet = detect_mallet(frame)
    
    if mallet:
        print(mallet)
        mallet_x, mallet_y, mallet_radius = mallet
        # Draw mallet
        cv2.circle(frame, (mallet_x,mallet_y), 10, (0,0,255), 10)

        # Check for puck collision with mallet
        check_collision(np.array([10, 20]))

    # Update puck position and movement
    update_puck()

    # Draw puck
    cv2.circle(frame, tuple(puck_pos.astype(int)), 20, (255,0,0), 5)

    # Draw goals (optional)
    cv2.rectangle(frame, (0, (480 // 2) - (120 // 2)), (10, (480 // 2) + (120 // 2)), (255,255,255),20)
    cv2.rectangle(frame, (640-10, (480 // 2) - (120 // 2)), (640, (480 // 2) + (120 // 2)), (255,255,255), 20)
    while(difference<=duration):
        ret, frame = cap.read()
        cv2.putText(frame,str(difference),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
 # Display the game frame
    cv2.imshow("Air Hockey", frame)
    difference=(datetime.now()-start_time).seconds

    if cv2.waitKey(10) & 0xFF == ord('r'):
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        quality=1
        break
cap.release()
cv2.destroyAllWindows()
