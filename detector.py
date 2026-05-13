import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
import time
from datetime import datetime

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Text to Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# State
current_gesture = ""
last_spoken = ""
last_spoken_time = 0
speak_cooldown = 2.0
stable_count = 0
stable_threshold = 15
sentence = []
gesture_history = []   # Last 5 gestures
is_speaking = False

# Log file
log_file = f"gesture_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def speak(text):
    global is_speaking
    def run():
        global is_speaking
        is_speaking = True
        engine.say(text)
        engine.runAndWait()
        is_speaking = False
    threading.Thread(target=run, daemon=True).start()

def get_landmarks(hand_results):
    data = []
    for hand_landmarks in hand_results[:2]:  # Process up to 2 hands
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    if len(hand_results) == 1:
        data.extend([0.0] * 63)
    return data

def log_gesture(gesture):
    """Save detected gesture to log file"""
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} - {gesture}\n")

def draw_ui(frame, gesture, confidence, sentence, history, speaking):
    h, w, _ = frame.shape

    # ── Top bar ──────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 20), -1)

    # Gesture name
    color = (0, 255, 0) if gesture else (100, 100, 100)
    label = gesture if gesture else "No Gesture"
    cv2.putText(frame, label, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

    # Confidence bar background
    cv2.rectangle(frame, (w-230, 15), (w-20, 40), (60, 60, 60), -1)
    if confidence > 0:
        conf_w = int(confidence * 210)
        bar_color = (0, 255, 0) if confidence > 0.85 else (0, 165, 255)
        cv2.rectangle(frame, (w-230, 15),
                     (w-230+conf_w, 40), bar_color, -1)
    cv2.putText(frame, f"{confidence*100:.0f}%",
                (w-225, 33), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    # Speaking indicator
    if speaking:
        cv2.circle(frame, (w-15, 55), 8, (0, 255, 255), -1)
        cv2.putText(frame, "Speaking",
                    (w-100, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)

    # ── History panel (right side) ───────────────
    cv2.rectangle(frame, (w-160, 70), (w, 250), (20, 20, 20), -1)
    cv2.putText(frame, "History:",
                (w-150, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)
    for i, g in enumerate(history[-5:][::-1]):
        alpha = 255 - (i * 40)
        cv2.putText(frame, f"• {g}",
                    (w-145, 120 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (alpha, alpha, alpha), 1)

    # ── Sentence bar (bottom) ────────────────────
    cv2.rectangle(frame, (0, h-70), (w, h), (20, 20, 20), -1)
    sentence_text = " ".join(sentence[-6:])
    cv2.putText(frame, f"{sentence_text}",
                (10, h-40), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 0), 2)
    cv2.putText(frame, "S=Speak | C=Clear | Q=Quit",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150, 150, 150), 1)

print("=== Sign Language Detector — Final Version ===")
print(f"📝 Logging to: {log_file}")
print("S = Speak sentence | C = Clear | Q = Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    detected_gesture = ""
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

        landmark_data = get_landmarks(result.multi_hand_landmarks)
        prediction = model.predict([landmark_data])[0]
        proba = model.predict_proba([landmark_data])[0]
        confidence = max(proba)

        if confidence > 0.75:
            detected_gesture = prediction

            if detected_gesture == current_gesture:
                stable_count += 1
            else:
                stable_count = 0
                current_gesture = detected_gesture

            current_time = time.time()
            if (stable_count == stable_threshold and
                    detected_gesture != last_spoken and
                    current_time - last_spoken_time > speak_cooldown):

                speak(detected_gesture)
                sentence.append(detected_gesture)
                gesture_history.append(detected_gesture)
                log_gesture(detected_gesture)
                last_spoken = detected_gesture
                last_spoken_time = current_time
                print(f"🤟 {detected_gesture} ({confidence*100:.1f}%)")

    else:
        stable_count = 0
        current_gesture = ""

    draw_ui(frame, detected_gesture, confidence,
            sentence, gesture_history, is_speaking)

    cv2.imshow("Sign Language Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if sentence:
            full = " ".join(sentence)
            speak(full)
            print(f"🔊 Speaking: {full}")

    if key == ord('c'):
        sentence = []
        last_spoken = ""
        print("🗑️  Cleared")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Session saved to {log_file}")