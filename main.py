import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

names = ["Aahad"]

camera = cv2.VideoCapture(0)

window_name = "AI Face Recognition"

try:
    if not camera.isOpened():
        print("Camera not working")
        raise SystemExit

    cv2.namedWindow(window_name)

    while True:
        # Check karo ki window abhi bhi open hai
        try:
            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break
        except cv2.error:
            break

        success, frame = camera.read()

        if not success:
            print("Camera not working")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]

            label, confidence = recognizer.predict(face)

            if confidence < 60 and label < len(names):
                name = names[label]
            else:
                name = "Unknown"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        cv2.imshow(window_name, frame)

        # Q dabane par bhi camera close hoga
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Camera ko properly release karo
    camera.release()
    cv2.destroyAllWindows()

    # OpenCV ki pending window events process karo
    cv2.waitKey(1)

    print("AI Face Recognition stopped.")