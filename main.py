import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# Constants
WIDTH, HEIGHT = 1280, 720
MALLET_RADIUS = 30
PUCK_RADIUS = 20
SPEED_LIMIT = 15
GOAL_SIZE = 200
DURATION = 120

# Colors
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Physics variables for puck
puck_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float32)
puck_vel = np.array([0, 0], dtype=np.float32)

# Game state variables
score_player1 = 0
score_player2 = 0

def update_puck():
    global puck_pos, puck_vel, score_player1, score_player2

    # Update position
    new_pos = puck_pos + puck_vel

    # Wall collisions with proper bouncing
    # Vertical walls (top and bottom)
    if new_pos[1] - PUCK_RADIUS <= 0:
        new_pos[1] = PUCK_RADIUS
        puck_vel[1] = abs(puck_vel[1])  # Bounce downward
    elif new_pos[1] + PUCK_RADIUS >= HEIGHT:
        new_pos[1] = HEIGHT - PUCK_RADIUS
        puck_vel[1] = -abs(puck_vel[1])  # Bounce upward

    # Horizontal walls (left and right)
    if new_pos[0] - PUCK_RADIUS <= 0:
        # Check if it's a goal
        if (HEIGHT // 2 - GOAL_SIZE // 2) <= new_pos[1] <= (HEIGHT // 2 + GOAL_SIZE // 2):
            score_player2 += 1
            reset_puck()
            return
        new_pos[0] = PUCK_RADIUS
        puck_vel[0] = abs(puck_vel[0])  # Bounce right
    elif new_pos[0] + PUCK_RADIUS >= WIDTH:
        # Check if it's a goal
        if (HEIGHT // 2 - GOAL_SIZE // 2) <= new_pos[1] <= (HEIGHT // 2 + GOAL_SIZE // 2):
            score_player1 += 1
            reset_puck()
            return
        new_pos[0] = WIDTH - PUCK_RADIUS
        puck_vel[0] = -abs(puck_vel[0])  # Bounce left

    puck_pos[:] = new_pos

    # Clamp speed
    speed = np.linalg.norm(puck_vel)
    if speed > SPEED_LIMIT:
        puck_vel = (puck_vel / speed) * SPEED_LIMIT

def reset_puck():
    global puck_pos, puck_vel
    puck_pos[:] = [WIDTH // 2, HEIGHT // 2]
    angle = np.random.uniform(0, 2 * np.pi)
    speed = 5
    puck_vel[:] = [speed * np.cos(angle), speed * np.sin(angle)]

def check_collision(mallet_pos, mallet_vel):
    global puck_pos, puck_vel

    # Vector from mallet to puck
    to_puck = puck_pos - mallet_pos
    distance = np.linalg.norm(to_puck)

    if distance <= (MALLET_RADIUS + PUCK_RADIUS):
        # Normalize the direction vector
        collision_dir = to_puck / distance if distance > 0 else np.array([1, 0])

        # Calculate relative velocity
        rel_vel = puck_vel - mallet_vel

        # Calculate the velocity component along the collision direction
        vel_along_collision = np.dot(rel_vel, collision_dir)

        # Only bounce if the objects are moving toward each other
        if vel_along_collision < 0:
            # Elastic collision with full energy transfer
            new_vel = collision_dir * abs(vel_along_collision)
            puck_vel = new_vel + mallet_vel  # Full transfer of mallet velocity

            # Move puck outside collision radius to prevent sticking
            separation = (MALLET_RADIUS + PUCK_RADIUS - distance + 1)
            puck_pos[:] = mallet_pos + collision_dir * (MALLET_RADIUS + PUCK_RADIUS + 1)

def assign_hands(results):
    if not results.multi_hand_landmarks:
        return None, None

    hands_data = []
    for hand_landmarks in results.multi_hand_landmarks:
        x = hand_landmarks.landmark[0].x * WIDTH
        hands_data.append((x, hand_landmarks))

    if len(hands_data) == 2:
        hands_data.sort(key=lambda x: x[0])
        return hands_data[0][1], hands_data[1][1]
    elif len(hands_data) == 1:
        x_pos = hands_data[0][0]
        if x_pos < WIDTH / 2:
            return hands_data[0][1], None
        else:
            return None, hands_data[0][1]
    return None, None

def draw_game_over(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(frame, "Game Over!", (WIDTH // 2 - 200, HEIGHT // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)

    winner_text = "It's a Tie!" if score_player1 == score_player2 else f"Player {'1' if score_player1 > score_player2 else '2'} Wins!"
    cv2.putText(frame, winner_text, (WIDTH // 2 - 150, HEIGHT // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.putText(frame, "Press 'N' for new game or 'Q' to quit",
                (WIDTH // 2 - 250, HEIGHT // 2 + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

cap = cv2.VideoCapture(0)
start_time = datetime.now()
mallet_pos_player1 = np.array([WIDTH // 4, HEIGHT // 2], dtype=np.float32)
mallet_pos_player2 = np.array([3 * WIDTH // 4, HEIGHT // 2], dtype=np.float32)
prev_mallet_pos_player1 = mallet_pos_player1.copy()
prev_mallet_pos_player2 = mallet_pos_player2.copy()
game_over = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not accessible")
        break

    elapsed_time = (datetime.now() - start_time).seconds
    remaining_time = max(DURATION - elapsed_time, 0)

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if not game_over:
        # Store previous mallet positions
        prev_mallet_pos_player1 = mallet_pos_player1.copy()
        prev_mallet_pos_player2 = mallet_pos_player2.copy()

        # Process hand tracking
        hand1, hand2 = assign_hands(results)

        if hand1 is not None:
            palm_center = hand1.landmark[0]
            mallet_pos_player1 = np.array([palm_center.x * WIDTH, palm_center.y * HEIGHT], dtype=np.float32)
            mp_drawing.draw_landmarks(frame, hand1, mp_hands.HAND_CONNECTIONS)

        if hand2 is not None:
            palm_center = hand2.landmark[0]
            mallet_pos_player2 = np.array([palm_center.x * WIDTH, palm_center.y * HEIGHT], dtype=np.float32)
            mp_drawing.draw_landmarks(frame, hand2, mp_hands.HAND_CONNECTIONS)

        # Calculate mallet velocities
        mallet_vel_player1 = (mallet_pos_player1 - prev_mallet_pos_player1)
        mallet_vel_player2 = (mallet_pos_player2 - prev_mallet_pos_player2)

        # Draw mallets
        cv2.circle(frame, tuple(mallet_pos_player1.astype(int)), MALLET_RADIUS, RED, 5)
        cv2.circle(frame, tuple(mallet_pos_player2.astype(int)), MALLET_RADIUS, BLUE, 5)

        # Check collisions and update physics
        check_collision(mallet_pos_player1, mallet_vel_player1)
        check_collision(mallet_pos_player2, mallet_vel_player2)
        update_puck()

    # Draw game elements
    cv2.circle(frame, tuple(puck_pos.astype(int)), PUCK_RADIUS, WHITE, 5)
    cv2.rectangle(frame, (0, (HEIGHT // 2) - (GOAL_SIZE // 2)), (10, (HEIGHT // 2) + (GOAL_SIZE // 2)), WHITE, 20)
    cv2.rectangle(frame, (WIDTH - 10, (HEIGHT // 2) - (GOAL_SIZE // 2)), (WIDTH, (HEIGHT // 2) + (GOAL_SIZE // 2)), WHITE, 20)
    cv2.line(frame, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), GREEN, 5)
    cv2.circle(frame, (WIDTH // 2, HEIGHT // 2), 100, GREEN, 5)
    cv2.circle(frame, (WIDTH // 2, HEIGHT // 2), 20, WHITE, -1)
    cv2.circle(frame, (0, HEIGHT // 2), GOAL_SIZE // 2, YELLOW, 5)
    cv2.circle(frame, (WIDTH, HEIGHT // 2), GOAL_SIZE // 2, YELLOW, 5)

    # Display time and score
    cv2.putText(frame, f"Time Left: {remaining_time}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Player 1: {score_player1}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Player 2: {score_player2}", (WIDTH - 220, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2, cv2.LINE_AA)

    if remaining_time <= 0:
        game_over = True
        frame = draw_game_over(frame)

    cv2.imshow("Air Hockey", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n') and game_over:
        game_over = False
        score_player1, score_player2 = 0, 0
        start_time = datetime.now()
        reset_puck()

cap.release()
cv2.destroyAllWindows()

