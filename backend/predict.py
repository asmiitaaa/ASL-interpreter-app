import cv2
import pickle
import numpy as np
from utils.hand_detector import HandDetector

MODEL_FILE = 'models/asl_model.pkl'

# load the trained model
with open(MODEL_FILE, 'rb') as f:
    data = pickle.load(f)#loading the pickled model file 
    model = data['model']#in the pickle, we split the csv data into model| and labels, so that is what we have here 
    labels = data['labels']
print(f"signs the model knows: {labels}")
#creating detector and webcam objects that we will use, this is why detector is a util, we use instances of it everywehere 
detector = HandDetector()
cap = cv2.VideoCapture(0)
print("Show your hand — press Q to quit")
while True:
    ret, frame = cap.read()#read defined by opencv 
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame, results = detector.find_hands(frame)
    landmarks = detector.get_landmarks(results)
    if landmarks:#if the points are detected 
        # flatten 21 points to 63 numbers
        normalized = detector.normalize_landmarks(landmarks)#we normalize them with respect to the wrist 
        flat = []
        for point in normalized:
            for coord in point:
                flat.append(coord)
        # feed into model — returns predicted sign
        prediction = model.predict([flat])[0]
        confidence = max(model.predict_proba([flat])[0]) * 100#how many of the 100 decision trees support the ans 
        # display prediction on screen
        cv2.putText(
            frame,
            f"Sign: {prediction} ({confidence:.1f}%)",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    cv2.imshow("ASL Predictor", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()