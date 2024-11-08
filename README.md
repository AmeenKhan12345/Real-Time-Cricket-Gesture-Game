# Real-Time Cricket Gesture Game

This is a real-time cricket game where you play against the machine using hand gestures. Show your hand in the green box on the webcam feed to score runs, bowl, or participate in a toss. The machine uses gesture recognition to play against you, making the experience fun and interactive.

## Features

- Toss the game with hand gestures to decide who bats or bowls first.
- Use hand gestures to score runs or bowl, making it a real-time, dynamic experience.
- Enjoy the thrill of a cricket game where you play against a machine player.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/AmeenKhan12345/Real-Time-Cricket-Gesture-Game.git
    cd Real-Time-Cricket-Gesture-Game/Odd-Eve_Game
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the trained model file (`your_model.h5`) in the `Odd-Eve_Game` directory.

## Usage

1. Run the game using:
    ```bash
    python Cricket_Finger_Gesture_Game.py
    ```

2. Place your hand in the green box displayed on the webcam feed:
   - For the **toss**: Show any hand gesture and press Enter.
   - For **batting**: Show finger signs to indicate runs.
   - For **bowling**: Predict the machine's score by showing different hand signs.
   
3. The game will proceed automatically based on the runs, outs, and innings.

## Requirements

- Python 3.8+
- Webcam

## How to Play

1. **Start the game** and select “Odd” or “Even” for the toss.
2. **Win the toss** to decide if you want to bat or bowl first.
3. **Use finger signs** to indicate runs (batting) or to predict runs (bowling).
4. The **machine plays against you** by generating random runs.
5. The game ends when both innings are complete or if a player achieves the target score.

Enjoy playing this interactive, gesture-based cricket game!

---

**Note**: Ensure the `your_model.h5` file is present in the `Odd-Eve_Game` directory. Update its path if necessary in `main.py`.

---

## License

This project is licensed under the MIT License.
