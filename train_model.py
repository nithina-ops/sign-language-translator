import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Your 10 gestures
gestures = [
    "Hello", "Yes", "No", "Thanks", "Sorry",
    "Happy", "Sad", "Love", "Peace", "Stop"
]

data_dir = "data"
X = []  # landmark data
y = []  # labels

print("=== Loading Gesture Data ===\n")

# Load all gesture data
for gesture in gestures:
    path = os.path.join(data_dir, f"{gesture}.npy")
    if os.path.exists(path):
        samples = np.load(path)
        X.extend(samples)
        y.extend([gesture] * len(samples))
        print(f"✅ Loaded {len(samples)} samples for '{gesture}'")
    else:
        print(f"❌ Missing data for '{gesture}' — run collect_data.py first!")

print(f"\nTotal samples: {len(X)}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# Train Random Forest Classifier
print("\n=== Training Model ===")
model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    max_depth=6,        # Lower = less memorization
    min_samples_leaf=5, # Each leaf needs 5+ samples
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model trained!")

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Results ===")
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Save the model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved as gesture_model.pkl")
print("\n✅ Training complete! Run detector.py next!")