import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces_folder = "faces"

training_faces = []
training_labels = []
names = []

for person_id, person_name in enumerate(os.listdir(faces_folder)):
    person_folder = os.path.join(faces_folder, person_name)

    if not os.path.isdir(person_folder):
        continue

    names.append(person_name)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in detected_faces:
            face = gray[y:y + h, x:x + w]

            training_faces.append(face)
            training_labels.append(person_id)

print("Training faces:", len(training_faces))
print("Names:", names)

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    training_faces,
    np.array(training_labels)
)

recognizer.write("face_model.yml")

print("Training complete!")
print("Model saved as face_model.yml")