import pygame
import cv2
import time
import random
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the gesture recognition model
model = tf.keras.models.load_model('C:/Users/ASUS/Documents/Odd-Eve_Game/model.h5')

# Initialize Pygame
pygame.init()

# Set up display
width, height = 1000, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Real-time Cricket Gesture Game (ODD-EVE)")

# Font for displaying text
font = pygame.font.SysFont(None, 55)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def capture_hand_from_frame(frame):
    # Convert frame color (BGR to RGB) as MediaPipe expects RGB input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box of the hand
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = y_max = 0
            
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
            
            # Crop the hand region
            hand_image = frame[y_min:y_max, x_min:x_max]
            hand_image = cv2.resize(hand_image, (256, 256))
            hand_image = np.expand_dims(hand_image, axis=0) / 255.0  # Normalize

            return hand_image  # Return the cropped and preprocessed hand image
    return None



# Function to draw text on the screen
def draw_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    window.blit(text_surface, (x, y))

# display_webcam_frame remains the same
def display_webcam_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        window.blit(frame_surface, (350, 200))

        # Draw green box for gesture area
        box_width, box_height = 256, 256
        box_x, box_y = 350 + (frame.shape[1] - box_width) // 2, 200 + (frame.shape[0] - box_height) // 2
        pygame.draw.rect(window, (0, 255, 0), pygame.Rect(box_x, box_y, box_width, box_height), 2)
        
    return frame

# Update capture_in_green_box
def capture_in_green_box(frame):
    hand_image = capture_hand_from_frame(frame)
    if hand_image is not None:
        return hand_image
    else:
        # Default handling if no hand detected (e.g., return a blank frame or handle error)
        return np.zeros((1, 256, 256, 3))

# Gesture value mapping
def get_gesture_value(gesture):
    return gesture + 1 if gesture < 9 else 10  # Maps 0-9 gestures to 1-10 runs

# Game logic for batting
# Refined user_batting to avoid lag with minimum updates
def user_batting(target_score=None):
    user_score, is_out = 0, False
    
    while running and not is_out:
        window.fill((0, 0, 0))
        draw_text("Batting!!", font, (255, 255, 255), 100, 10)
        draw_text(f"Target:  {target_score}", font, (255, 255, 255), 100, 40)
        draw_text(f"Your Score: {user_score}", font, (255, 255, 255), 100, 100)
        draw_text("Show your gesture and press Enter to bat!", font, (255, 255, 255), 150, 150)
        
        # Display webcam feed
        frame = display_webcam_frame()
        pygame.display.update()  # Update display only once per loop for smoothness
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # Capture and predict the gesture
                cropped_image = capture_in_green_box(frame)
                predictions = model.predict(cropped_image)
                predicted_gesture = get_gesture_value(np.argmax(predictions))

                # Machine's random bowling number
                machine_number = random.randint(1, 10)

                # Display both the predicted gesture and the machine's runs
                draw_text(f"Your Gesture Run: {predicted_gesture}", font, (255, 255, 255), 50, 250)
                draw_text(f"Machine's Runs: {machine_number}", font, (255, 255, 255), 50, 350)
                
                if predicted_gesture == machine_number:
                    draw_text("You're Out!", font, (255, 0, 0), 400, 400)
                    pygame.display.update()
                    is_out = True
                else:
                    user_score += predicted_gesture
                    if target_score and user_score >= target_score:
                        return user_score, True
                    
                pygame.display.update()  # Refresh screen after each prediction
                pygame.time.delay(1500)  # Short delay for readability
    
    return user_score, is_out


# Game logic for bowling
def user_bowling(target_score=None):
    machine_score, is_out = 0, False
    
    while running and not is_out:
        window.fill((0, 0, 0))
        draw_text("Bowling!!", font, (255, 255, 255), 100, 10)
        draw_text(f"Target:  {target_score}", font, (255, 255, 255), 100, 40)
        draw_text(f"Machine Score: {machine_score}", font, (255, 255, 255), 100, 100)
        draw_text("Show your gesture and press Enter to bowl!", font, (255, 255, 255), 150, 150)
        
        # Display webcam feed
        frame = display_webcam_frame()
        pygame.display.update()  # Update display only once per loop for smoothness
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # Capture and predict the gesture
                cropped_image = capture_in_green_box(frame)
                predictions = model.predict(cropped_image)
                predicted_gesture = get_gesture_value(np.argmax(predictions))

                # Machine's random batting run
                machine_number = random.randint(1, 10)
                
                # Display machine's run and user's gesture
                draw_text(f"Machine's Runs: {machine_number}", font, (255, 255, 255), 50, 250)
                draw_text(f"Your Gesture Run: {predicted_gesture}", font, (255, 255, 255), 50, 350)
                
                if predicted_gesture == machine_number:
                    draw_text("Machine is Out!", font, (255, 0, 0), 300, 350)
                    pygame.display.update()
                    is_out = True
                else:
                    machine_score += machine_number
                    if target_score and machine_score >= target_score:
                        return machine_score, True
                
                pygame.display.update()  # Refresh screen after each prediction
                pygame.time.delay(1500)
    
    return machine_score, is_out



# Declare winner at the end of the game
def declare_winner():
    window.fill((0, 0, 0))
    draw_text(f"Your Score: {user_score}", font, (255, 255, 255), 100, 100)
    draw_text(f"Machine's Score: {machine_score}", font, (255, 255, 255), 100, 150)
    result_text = "Congrats,You Won!" if user_score > machine_score else "OOPS, Machine Wins!" if user_score < machine_score else "It's a Draw!"
    draw_text(result_text, font, (255, 255, 0), 200, 300)
    pygame.display.update()
    time.sleep(3)
    global running
    running = False

# Main game loop variables
running, game_stage = True, "start"
user_choice, toss_winner, bat_or_bowl_choice = None, None, None
user_score, machine_score, target_score = 0, 0, None
user_inning_complete, machine_inning_complete = False, False

# Main game loop
while running:
    window.fill((0, 0, 0))

    # Game stages handling
    if game_stage == "start":
        draw_text("Welcome to Odd-Eve Game", font, (255, 255, 255), 100, 200)
        draw_text("Press any key to begin...", font, (255, 255, 255), 100, 300)

    elif game_stage == "odd_or_even":
        draw_text("ODD or EVE?", font, (255, 255, 255), 100, 200)
        draw_text("Press 'o' for ODD and 'e' for EVE", font, (255, 255, 255), 100, 300)
        if user_choice:
            draw_text(f"Chosen: {user_choice}", font, (255, 255, 255), 100, 400)

    elif game_stage == "chosen":
        draw_text(f"Chosen: {user_choice}", font, (255, 255, 255), 100, 300)
        pygame.display.update()
        time.sleep(2)
        game_stage = "rules"

    elif game_stage == "rules":
        draw_text("Game Rules:", font, (255, 255, 255), 100, 100)
        draw_text("1 run = Index finger, 2 runs = Index + Middle, etc.", font, (255, 255, 255), 100, 200)
        draw_text("6 runs = Thumb, 7 runs = Thumb + Index, etc.", font, (255, 255, 255), 100, 250)
        draw_text("10 runs = Fist", font, (255, 255, 255), 100, 300)
        draw_text("Press any key to continue...", font, (255, 255, 255), 100, 400)

    elif game_stage == "toss":
        draw_text("TOSS! Show gesture in the box", font, (255, 255, 255), 100, 50)
        draw_text("Press Enter when ready.", font, (255, 255, 255), 100, 100)
        frame = display_webcam_frame()
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    
                # Capture gesture and make a prediction
                cropped_image = capture_in_green_box(frame)
                predictions = model.predict(cropped_image)
                predicted_gesture = get_gesture_value(np.argmax(predictions))
                
                # Generate random toss value for the machine
                machine_toss = random.randint(1, 10)

                # Check if user wins the toss
                toss_sum = predicted_gesture + machine_toss
                user_wins_toss = (toss_sum % 2 == 0 and user_choice == "Even") or \
                                 (toss_sum % 2 != 0 and user_choice == "Odd")

                toss_winner = "User" if user_wins_toss else "Machine"
                
                # Display toss results
                draw_text(f"You chose: {predicted_gesture}, Machine chose: {machine_toss}", font, (255, 255, 255), 100, 200)
                draw_text(f"{toss_winner} wins the toss!", font, (255, 255, 0), 100, 300)
                
                pygame.display.update()
                time.sleep(2)  # Brief pause to show result
                # Set next stage based on toss winner
                game_stage = "bat_or_bowl_user" if toss_winner == "User" else "bat_or_bowl_machine"


    elif game_stage == "bat_or_bowl_user":
        draw_text("Choose to Bat (press 'b') or Bowl (press 'k')?", font, (255, 255, 255), 100, 50)
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_b:
                    player_action = "bat"
                    game_stage = "user_batting"
                    break  # Exit the event loop once choice is made
                elif event.key == pygame.K_k:
                    player_action = "bowl"
                    game_stage = "user_bowling"
                    break  # Exit the event loop once choice is made

    elif game_stage == "bat_or_bowl_machine":
        machine_choice = random.choice(["Bat", "Bowl"])
        draw_text(f"Machine chooses to: {machine_choice}", font, (255, 255, 255), 100, 100)
        pygame.display.update()
        time.sleep(2)
        game_stage = "user_bowling" if machine_choice == "Bat" else "user_batting"

    # First inning: User or machine bats first based on toss
    if game_stage == "user_batting" and not user_inning_complete:
        user_score, is_out = user_batting()
        if is_out:
            user_inning_complete = True
            target_score = user_score + 1  # Set target for machine
            game_stage = "user_bowling"
        elif target_score and user_score >= target_score:
            # User reaches target score in second inning
            game_stage = "declare_winner"

    elif game_stage == "user_bowling" and not machine_inning_complete:
        machine_score, is_out = user_bowling()
        if is_out:
            machine_inning_complete = True
            target_score = machine_score + 1  # Set target for user
            game_stage = "user_batting"
        elif target_score and machine_score >= target_score:
            # Machine reaches target score in second inning
            game_stage = "declare_winner"

    # Second inning: Chasing the target score
    if game_stage == "user_batting" and machine_inning_complete:
        user_score, is_out = user_batting(target_score)
        if is_out or user_score >= target_score:
            game_stage = "declare_winner"  # End game if user meets or exceeds target

    elif game_stage == "user_bowling" and user_inning_complete:
        machine_score, is_out = user_bowling(target_score)
        if is_out or machine_score >= target_score:
            game_stage = "declare_winner"  # End game if machine meets or exceeds target

    elif game_stage == "declare_winner":
        declare_winner()

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if game_stage == "start":
                game_stage = "odd_or_even"
            elif game_stage == "odd_or_even":
                user_choice = "Odd" if event.key == pygame.K_o else "Even"
                game_stage = "chosen"
            elif game_stage == "rules":
                game_stage = "toss" 
            elif game_stage == "bat_or_bowl_user":
                bat_or_bowl_choice = "Bat" if event.key == pygame.K_b else "Bowl"
                game_stage = "user_batting" if bat_or_bowl_choice == "Bat" else "user_bowling"
    
    pygame.display.update()

# Cleanup
cap.release()
pygame.quit()
