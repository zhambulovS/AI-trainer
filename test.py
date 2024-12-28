import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils import calculate_angle, detection_body_part
from body_part_angle import BodyPartAngle
from types_of_excercise import TypeOfExercise

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_image(image_path, pose_model):
    """Processes a single image and calculates pose angles."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        angles = BodyPartAngle(landmarks)

        return {
            "left_arm_angle": angles.angle_of_the_left_arm(),
            "right_arm_angle": angles.angle_of_the_right_arm(),
            "left_leg_angle": angles.angle_of_the_left_leg(),
            "right_leg_angle": angles.angle_of_the_right_leg(),
            "neck_angle": angles.angle_of_the_neck(),
            "abdomen_angle": angles.angle_of_the_abdomen()
        }
    return None

def main():
    # Load CSV file containing image paths and labels
    csv_file = "pose_data.csv"  # Replace with your CSV file name
    data = pd.read_csv(csv_file)

    image_folder = "images/"  # Replace with your image folder path
    results_list = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for index, row in data.iterrows():
            image_path = os.path.join(image_folder, row['image_name'])

            if os.path.exists(image_path):
                angles = process_image(image_path, pose)

                if angles:
                    angles["label"] = row['label']
                    results_list.append(angles)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # Split data into features and labels
    X = results_df.drop(columns=["label", "image_name"], errors='ignore')
    y = results_df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a machine learning model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Save the model (optional)
    import joblib
    joblib.dump(clf, "exercise_classifier_model.pkl")

if __name__ == "__main__":
    main()
