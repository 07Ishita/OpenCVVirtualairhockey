import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# Constants
WIDTH, HEIGHT = 640, 480  # Screen size
MALLET_RADIUS = 16       # Radius of mallet
PUCK_RADIUS = 8          # Radius of puck
SPEED_LIMIT = 15          # Puck speed limit
GOAL_SIZE = 200           # Size of the goals
GAME_DURATION = 120       # GAME TIMING

# Colors
WHITE = (255, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Physics variables for puck
puck_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float32)  # Position of puck
puck_vel = np.random.uniform(-1, 1, size=2) * SPEED_LIMIT         # Starting velocity
Player1_score = 0
Player2_score = 0

# Function to update puck physics (movement and collision)
def update_puck():
    global puck_pos, puck_vel, Player1_score, Player2_score
    puck_pos += puck_vel

    # Collision with walls (excluding goal areas)
    if puck_pos[1] - PUCK_RADIUS <= 0 or puck_pos[1] + PUCK_RADIUS >= HEIGHT:
        puck_vel[1] *= -1  # Bounce vertically
    if puck_pos[0] - PUCK_RADIUS <= 0 and (HEIGHT // 2 - GOAL_SIZE // 2) <= puck_pos[1] <= (HEIGHT // 2 + GOAL_SIZE // 2):
        Player1_score += 1
        reset_puck()
    elif puck_pos[0] + PUCK_RADIUS >= WIDTH and (HEIGHT // 2 - GOAL_SIZE // 2) <= puck_pos[1] <= (HEIGHT // 2 + GOAL_SIZE // 2):
        Player2_score += 1
        reset_puck() 
    elif puck_pos[0] - PUCK_RADIUS <= 0 or puck_pos[0] + PUCK_RADIUS >= WIDTH:
        puck_vel[0] *= -1  # Bounce horizontally    

    # Clamp speed
    speed = np.linalg.norm(puck_vel)
    if speed > SPEED_LIMIT:
        puck_vel = (puck_vel / speed) * SPEED_LIMIT 

def reset_puck():
    global puck_pos, puck_vel  
    puck_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float32)  # Position of puck
    puck_vel = np.random.uniform(-1, 1, size=2) * SPEED_LIMIT          # Starting velocity   

# Function to check for collision with the mallet
def check_collision(mallet_positions):
    global puck_pos, puck_vel
    for mallet_pos in mallet_positions:
        dist = np.linalg.norm(puck_pos - mallet_pos)
        if dist <= MALLET_RADIUS + PUCK_RADIUS:
            direction = (puck_pos - mallet_pos) / dist
            puck_vel = direction * SPEED_LIMIT  # Reflect puck with new velocity

# Function to control mallet with hands
def control_mallet_with_hands(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    mallet_positions = []  # To store positions of both hands
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand_landmarks.landmark[8]
            mallet_x = int(index_finger_tip.x * frame.shape[1])
            mallet_y = int(index_finger_tip.y * frame.shape[0])
            mallet_pos = np.array([mallet_x, mallet_y])
            mallet_positions.append(mallet_pos)
            cv2.circle(frame, (mallet_x, mallet_y), MALLET_RADIUS, RED, 60)
    
    return mallet_positions, frame

def display_start_menu(frame):
    cv2.putText(frame, "Air Hockey Game", (WIDTH // 2 - 150, HEIGHT // 2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)
    cv2.putText(frame, "Press 's' to Start", (WIDTH // 2 - 130, HEIGHT // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
    cv2.putText(frame, "Press 'q' to Quit", (WIDTH // 2 - 130, HEIGHT // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)

def display_game_over(frame):
     global Player1_score, England_score
     winner = "It's a Draw!"
     if Player1_score > Player2_score:
        winner = "Player1 Wins!"
     elif Player2_score > Player2_score:
        winner = "Player2 Wins!"

    # Display the game over message and winner
     cv2.putText(frame, "Game Over", (WIDTH // 2 - 100, HEIGHT // 2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 3)
     cv2.putText(frame, f"Final Score: Player1 {Player1_score} - Player2 {Player2_score}", (WIDTH // 2 - 200, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
     cv2.putText(frame, winner, (WIDTH // 2 - 100, HEIGHT // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 2)
     cv2.putText(frame, "Press 'r' to Restart", (WIDTH // 2 - 130, HEIGHT // 2 + 80), cv2.FONT_HERSHEY_SIMPLEX,1,GREEN,2)
# Main loop for the game
duration = 120
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
start_time = None
game_started = False

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    if not game_started:
        display_start_menu(frame)
        if cv2.waitKey(10) & 0xFF == ord('s'):
            start_time = datetime.now()
            game_started = True
        elif cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        difference = (datetime.now() - start_time).seconds
        if difference >= duration:
            display_game_over(frame)
            if cv2.waitKey(10) & 0xFF == ord('r'):
                Player1_score, Player2_score = 0, 0
                start_time = datetime.now()
            elif cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            mallet_pos, frame = control_mallet_with_hands(frame)
            update_puck()
            if mallet_pos is not None:
                check_collision(mallet_pos)

            # Draw puck and goals
            cv2.circle(frame, tuple(puck_pos.astype(int)), PUCK_RADIUS, BLUE, 40)
            cv2.rectangle(frame, (0, (HEIGHT // 2) - (GOAL_SIZE // 2)), (10, (HEIGHT // 2) + (GOAL_SIZE // 2)), WHITE, 20)
            cv2.rectangle(frame, (WIDTH - 10, (HEIGHT // 2) - (GOAL_SIZE // 2)), (WIDTH, (HEIGHT // 2) + (GOAL_SIZE // 2)), WHITE, 20)
            
            # Display timer and scores
            cv2.putText(frame, f"Time: {duration - difference}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
            cv2.putText(frame, f"Player1 : {Player1_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Player2 : {Player2_score}", (WIDTH - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Air Hockey", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
