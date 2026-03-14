import cv2
from utils.hand_detector import HandDetector 

detector=HandDetector()#creates a handdetector object
cap=cv2.VideoCapture(0)#tells opencv to open the webcame 0- default camera 

print("Camera open- show your hand. Press Q to quit")

while True:#runs till we quit
    ret, frame=cap.read()

    #ret-true if the frame was read successfully 
    if not ret:
        print("Couldn't read from camera")
        break

    frame=cv2.flip(frame,1)#flips horizontally so it feels natural, like looking in a mirror

    frame, results=detector.find_hands(frame)#sends the frame to the handdetector
    landmarks=detector.get_landmarks(results)#extracts the 21 xyz points from the results 

    if landmarks:#if this list is not empty, a hand was detected 
        flat=detector.flatten_landmarks(landmarks)
        print(f"Hand detected! {len(landmarks)} points, {len(flat)} numbers")
        index_tip=landmarks[8]
        print(f"Index tip -> x:{index_tip[0]:.2f}   y:{index_tip[1]:.2f}  z:{index_tip[2]:.2f}")#round to two decimal places and quit
    else:
        print("No hand detected")

    cv2.imshow("ASL Detection Test", frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


