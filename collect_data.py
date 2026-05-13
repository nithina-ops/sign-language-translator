import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Your 10 gestures
gestures = [
    "Hello", "Yes", "No", "Thanks", "Sorry",
    "Happy", "Sad", "Love", "Peace", "Stop"
]

SAMPLES_PER_GESTURE = 100
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def get_landmarks(hand_results):
    """Extract 126 values — both hands (pad with zeros if 1 hand)"""
    data = []
    for hand_landmarks in hand_results:
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    
    # If only 1 hand detected, pad with zeros for second hand
    if len(hand_results) == 1:
        data.extend([0.0] * 63)
    
    return data  # Always 126 values

print("=== Data Collection Started ===")
print(f"Collecting {SAMPLES_PER_GESTURE} samples for each gesture")
print("Press SPACE to start collecting each gesture")
print("Press Q to quit\n")

for gesture in gestures:
    samples = []
    collecting = False
    count = 0

    print(f"\n👉 Get ready for: '{gesture}'")
    print(f"   Do the sign and press SPACE to start collecting...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Status display
        status_color = (0, 255, 0) if collecting else (0, 165, 255)
        status_text = f"COLLECTING {count}/{SAMPLES_PER_GESTURE}" if collecting else "READY - Press SPACE"

        # Background box for text
        cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)

        # Gesture name
        cv2.putText(frame, f"Gesture: {gesture}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        # Status
        cv2.putText(frame, status_text,
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, status_color, 2)

        # Progress bar
        if collecting:
            progress = int((count / SAMPLES_PER_GESTURE) * 620)
            cv2.rectangle(frame, (10, 85), (10 + progress, 95),
                         (0, 255, 0), -1)

        if result.multi_hand_landmarks:
            # Draw landmarks on all detected hands
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            if collecting:
                # Combine both hands into one sample
                landmark_data = get_landmarks(result.multi_hand_landmarks)
                samples.append(landmark_data)
                count += 1

                if count >= SAMPLES_PER_GESTURE:
                    save_path = os.path.join(data_dir, f"{gesture}.npy")
                    np.save(save_path, np.array(samples))
                    print(f"   ✅ Saved {SAMPLES_PER_GESTURE} samples for '{gesture}'")
                    cv2.rectangle(frame, (100, 180), (540, 300),
                                 (0, 0, 0), -1)
                    cv2.putText(frame, f"'{gesture}' DONE!",
                                (150, 250), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 0), 3)
                    cv2.imshow("Data Collection", frame)
                    cv2.waitKey(1500)
                    collecting = False
                    break
        else:
            cv2.putText(frame, "No hand detected",
                        (200, 260), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and not collecting and count == 0:
            collecting = True
            print(f"   📸 Collecting samples for '{gesture}'...")

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Move to next gesture when done
        if count >= SAMPLES_PER_GESTURE:
            break

print("\n🎉 All gestures collected successfully!")
print("Run train_model.py next!")

cap.release()
cv2.destroyAllWindows()