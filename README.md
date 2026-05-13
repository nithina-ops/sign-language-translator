


# Real-Time Sign Language Translator

A two-hand gesture recognition system using MediaPipe and Machine Learning.

## Features
- Detects 10 custom hand gestures in real time
- Two-hand landmark detection (126 features)
- 98% model accuracy using Random Forest
- Text-to-speech output
- Sentence builder with gesture history
- Auto gesture logging

## Tech Stack
Python | MediaPipe | OpenCV | scikit-learn | NumPy | pyttsx3

## Project Structure
sign_language/
├── collect_data.py    # Collect gesture training data
├── train_model.py     # Train ML classifier
├── detector.py        # Real-time detection
├── data/              # Gesture landmark data
└── gesture_model.pkl  # Trained model

## How to Run

### 1. Install dependencies
pip install numpy==1.26.4 opencv-python==4.8.1.78 mediapipe==0.10.9 scikit-learn pyttsx3

### 2. Collect gesture data
python collect_data.py

### 3. Train model
python train_model.py

### 4. Run detector
python detector.py

## Controls
| Key | Action |
|-----|--------|
| S   | Speak full sentence |
| C   | Clear sentence |
| Q   | Quit |

## Results
- Model Accuracy: 98%
- Gestures: Hello, Yes, No, Thanks, Sorry, Happy, Sad, Love, Peace, Stop
