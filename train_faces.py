# train_faces.py
from utils import load_known_faces, save_encodings

encodings, names = load_known_faces()
save_encodings(encodings, names)
print("Training completed! Face encodings saved.")

