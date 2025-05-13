# utils.py
import os
import face_recognition
import pickle
from datetime import datetime

def load_known_faces(directory="known_faces"):
    known_encodings = []
    known_names = []
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def save_encodings(encodings, names, filename="encodings.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((encodings, names), f)

def load_encodings(filename="encodings.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

def mark_attendance(name, filename="attendance.csv"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"{name},{now}\n")

