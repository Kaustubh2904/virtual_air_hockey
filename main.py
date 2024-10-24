import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# Constants
WIDTH, HEIGHT = 1280, 720  # Updated screen size for larger view
MALLET_RADIUS = 30  # Radius of the player's mallet
PUCK_RADIUS = 20  # Radius of the puck
SPEED_LIMIT = 15  # Limit puck speed
GOAL_SIZE = 200  # Adjusted goal size for the larger window
DURATION = 120  # 2 minutes game time

# Colors for players and elements
RED = (0, 0, 255)  # Player 1 mallet
BLUE = (255, 0, 0)  # Player 2 mallet
WHITE = (255, 255, 255)  # Puck and goals
GREEN = (0, 255, 0)  # Midline and circles
YELLOW = (0, 255, 255)  # Corner arcs

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils

# Physics variables for puck
puck_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float32)
puck_vel = np.array([5, 5], dtype=np.float32)  # Starting velocity

# Game state variables
score_player1 = 0
score_player2 = 0

# Function to update puck physics (movement and collision)
def update_puck():
    global puck_pos, puck_vel, score_player1, score_player2

    # Update puck position
    puck_pos += puck_vel

    # Collision with walls (top and bottom)
    if puck_pos[1] - PUCK_RADIUS <= 0 or puck_pos[1] + PUCK_RADIUS >= HEIGHT:
        puck_vel[1] *= -1  # Bounce vertically

    # Goal detection for player 1 (right side goal)
    if puck_pos[0] - PUCK_RADIUS <= 0 and (HEIGHT // 2 - GOAL_SIZE // 2) <= puck_pos[1] <= (HEIGHT // 2 + GOAL_SIZE // 2):
        score_player2 += 1
        reset_puck()

    # Goal detection for player 2 (left side goal)
    if puck_pos[0] + PUCK_RADIUS >= WIDTH and (HEIGHT // 2 - GOAL_SIZE // 2) <= puck_pos[1] <= (HEIGHT // 2 + GOAL_SIZE // 2):
        score_player1 += 1
        reset_puck()

    # Collision with side walls
    if puck_pos[0] - PUCK_RADIUS <= 0 or puck_pos[0] + PUCK_RADIUS >= WIDTH:
        puck_vel[0] *= -1  # Bounce horizontally

    # Clamp speed
    speed = np.linalg.norm(puck_vel)
    if speed > SPEED_LIMIT:
        puck_vel = (puck_vel / speed) * SPEED_LIMIT

# Function to reset puck after a goal
def reset_puck():
    global puck_pos, puck_vel
    puck_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float32)
    puck_vel = np.random.uniform(-5, 5, size=2)  # Random direction after goal

# Function to check for collision with the mallet
def check_collision(mallet_pos, mallet_vel):
    global puck_pos, puck_vel

    dist = np.linalg.norm(puck_pos - mallet_pos)
    if dist <= MALLET_RADIUS + PUCK_RADIUS:
        # Calculate the collision direction
        direction = (puck_pos - mallet_pos) / dist

        # Add mallet velocity to puck velocity to make it realistic
        puck_vel = direction * SPEED_LIMIT + 0.5 * mallet_vel  # Adjust with mallet's influence

# Start video capture
cap = cv2.VideoCapture(0)

# Game variables
start_time = datetime.now()
mallet_pos_player1 = np.array([WIDTH // 4, HEIGHT // 2], dtype=np.float32)  # Initial mallet position for player 1
mallet_pos_player2 = np.array([3 * WIDTH // 4, HEIGHT // 2], dtype=np.float32)  # Initial mallet position for player 2
mallet_vel_player1 = np.array([0, 0], dtype=np.float32)
mallet_vel_player2 = np.array([0, 0], dtype=np.float32)

# Assign a fixed hand to each player
player1_assigned = False
player2_assigned = False

# New game functionality
game_over = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not accessible")
        break

    # Calculate time difference for the game duration
    elapsed_time = (datetime.now() - start_time).seconds

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Resize frame for consistency
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Loop through both hands and assign them to players based on initial detection
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            palm_center = hand_landmarks.landmark[0]
            current_mallet_pos = np.array([palm_center.x * WIDTH, palm_center.y * HEIGHT], dtype=np.float32)

            # Assign hand landmarks to fixed players
            if not player1_assigned:
                mallet_pos_player1 = current_mallet_pos
                player1_assigned = True
            elif not player2_assigned:
                mallet_pos_player2 = current_mallet_pos
                player2_assigned = True

            # Update mallet velocities (basic velocity approximation)
            if idx == 0:
                mallet_vel_player1 = current_mallet_pos - mallet_pos_player1
                mallet_pos_player1 = current_mallet_pos
            else:
                mallet_vel_player2 = current_mallet_pos - mallet_pos_player2
                mallet_pos_player2 = current_mallet_pos

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw mallets for both players
    cv2.circle(frame, tuple(mallet_pos_player1.astype(int)), MALLET_RADIUS, RED, 5)
    cv2.circle(frame, tuple(mallet_pos_player2.astype(int)), MALLET_RADIUS, BLUE, 5)

    # Check for puck collision with both mallets
    check_collision(mallet_pos_player1, mallet_vel_player1)
    check_collision(mallet_pos_player2, mallet_vel_player2)

    # Update puck position and movement
    update_puck()

    # Draw puck
    cv2.circle(frame, tuple(puck_pos.astype(int)), PUCK_RADIUS, WHITE, 5)

    # Draw goals
    cv2.rectangle(frame, (0, (HEIGHT // 2) - (GOAL_SIZE // 2)), (10, (HEIGHT // 2) + (GOAL_SIZE // 2)), WHITE, 20)
    cv2.rectangle(frame, (WIDTH - 10, (HEIGHT // 2) - (GOAL_SIZE // 2)), (WIDTH, (HEIGHT // 2) + (GOAL_SIZE // 2)), WHITE, 20)

    # Draw the midline and circles for aesthetics
    cv2.line(frame, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), GREEN, 5)  # Midline
    cv2.circle(frame, (WIDTH // 2, HEIGHT // 2), 100, GREEN, 5)  # Center circle
    cv2.circle(frame, (WIDTH // 2, HEIGHT // 2), 20, WHITE, -1)  # Puck circle
    cv2.circle(frame, (0, HEIGHT // 2), GOAL_SIZE // 2, YELLOW, 5)  # Left goal arc
    cv2.circle(frame, (WIDTH, HEIGHT // 2), GOAL_SIZE // 2, YELLOW, 5)  # Right goal arc

    # Display the time remaining and the score
    remaining_time = max(DURATION - elapsed_time, 0)
    cv2.putText(frame, f"Time Left: {remaining_time}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Player 1: {score_player1}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Player 2: {score_player2}", (WIDTH - 220, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2, cv2.LINE_AA)

    # Show the game frame
    cv2.imshow("Air Hockey", frame)

    if remaining_time <= 0:
        game_over = True

    # Handle game over and new game functionality
    if game_over:
        cv2.putText(frame, "Game Over!", (WIDTH // 2 - 100, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, "Press 'N' for new game or 'Q' to quit", (WIDTH // 2 - 200, HEIGHT // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the game
        break
    elif key == ord('n') and game_over:  # Start a new game
        game_over = False
        score_player1, score_player2 = 0, 0
        start_time = datetime.now()
        reset_puck()

cap.release()
cv2.destroyAllWindows()
