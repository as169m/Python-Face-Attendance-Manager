import face_recognition
import numpy as np
import json
import os

# Directory where your employee images are stored
IMAGES_DIR = "employee_images"  # Example: each file named "EMP001 - John Doe.jpg"
OUTPUT_JSON = "static/known_faces.json"

def generate_embeddings():
    known_faces = []
    
    for file_name in os.listdir(IMAGES_DIR):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        employee_name = os.path.splitext(file_name)[0]  # Extract "EMP001 - John Doe"
        image_path = os.path.join(IMAGES_DIR, file_name)

        print(f"Processing {employee_name}...")

        # Load image and compute face encoding
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"  [WARNING] No face found in {file_name}")
            continue

        face_descriptor = encodings[0]
        known_faces.append({
            "name": employee_name,
            "descriptor": face_descriptor.tolist()
        })

    # Save JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(known_faces, f)
    print(f"✅ Embeddings saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_embeddings()
