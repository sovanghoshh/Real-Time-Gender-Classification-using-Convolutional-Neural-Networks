import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model and face detector
model = load_model('C:\\Users\\sovan\\Downloads\\gender_classifier (1).keras')  # Use your downloaded model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_roi, (100, 100))  # Match your model's input shape
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        # Predict gender
        confidence = model.predict(input_tensor)[0][0]
        gender = 'Male' if confidence >= 0.5 else 'Female'
        color = (0, 255, 0) if gender == 'Male' else (0, 0, 255)

        # Display result
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f'{gender} {confidence:.2f}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Real-time Gender Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
