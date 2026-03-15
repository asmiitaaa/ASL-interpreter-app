import cv2
import csv
import os
import time
from utils.hand_detector import HandDetector#we are treating it like a package-hand detector 
#cv2-for webcam and window, time for the countdown 

SAMPLES_PER_SIGN=200#we take 100 samples for each sign that we record 

DATA_FILE='data/asl_data.csv'# we store the collected data in the collected_data.py file 

def setup():#no self, because it doesn't belong to a class 
    
    os.makedirs('data', exist_ok=True)#create the data/ folder if it doesn't exist 

    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', newline='') as f:#if it doesn't exist, create it 
            writer=csv.writer(f)
            header=['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x','y','z']]
            writer.writerow(header)
        print(f"Created new data file!:{DATA_FILE}")
    else:
        print(f"Added to existing data file!:{DATA_FILE}")
        
def countdown(window_name, frame, seconds=3):
    for i in range(seconds, 0, -1):#-1->cpunt backwards 
        display=frame.copy()#creates a copy of the frame called display, so that we display the text on the copy, not the original frame 
        cv2.putText(display,f"Get ready:{i}",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow(window_name, display)#displays the image in a window or in our screen-window_name, and display is the frame that we have to display inside that window 
        cv2.waitKey(1000)#wait 1s before moving onto the next one in countdown
#a frame is a single still image from a webcame


def save_sample(label, landmarks):#landmarks gives us the results for 21 points in our hand, we are now flattening the corrdinates of those 21 points 
    flat = []#flattens the coordinates, label is the sign for which gesture corresponds
    for point in landmarks:       # goes through each [x,y,z]
        for coord in point:       # goes through x, then y, then z
            flat.append(coord)    # adds each number to the list
    with open(DATA_FILE, 'a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([label]+flat)
#for each frame there is only one landmarks because we set num_hands=1, so everytime we run this, only one set of coordinates gets stored
'''
def collect_sign(sign_label, detector, cap):#cap is the webcame capture object, detector is the hand_detector object created in main()
    print(f"\nGet ready to sign:{sign_label}")
    print(f"Collecting {SAMPLES_PER_SIGN} samples ......")#we have set samples_per sign as 100 before itself 

    ret, frame=cap.read()
    frame=cv2.flip(frame,1)
    countdown("ASL Data Collector", frame)
    samples_collected =0

    while samples_collected < SAMPLES_PER_SIGN:
        ret,frame=cap.read()
        if not ret:
            break
        frame=cv2.flip(frame,1)#flips every frame horizontally so it feels like a mirror
        frame, results=detector.find_hands(frame)
        #sends the frame to the hand detector, the mdeiapipe scans it, draws the dots and liens and returns the results 
        landmarks=detector.get_landmarks(results)
        #extracts the 21xyz coordinate for the results 
        if landmarks:#only if a hand was detected, then save it
            save_sample(sign_label, landmarks)
            samples_collected+=1
        cv2.putText(
            frame,
            f"{sign_label}: {samples_collected}/{SAMPLES_PER_SIGN}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        cv2.imshow("ASL Data Collector", frame)
        cv2.waitKey(1)

    print(f"Done! Collected {samples_collected} samples for {sign_label}")
'''
def collect_sign(sign_label, detector, cap):
    print(f"\nGet ready to sign: {sign_label}")
    print(f"Press SPACE when ready to start collecting...")

    # wait for spacebar press before starting
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, results = detector.find_hands(frame)

        # show waiting message on screen
        cv2.putText(
            frame,
            f"Sign: {sign_label} — Press SPACE to start",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        cv2.imshow("ASL Data Collector", frame)

        # wait for spacebar (32 is the ascii code for space)
        key = cv2.waitKey(1)
        if key == 32:
            break

    # now start collecting
    samples_collected = 0

    while samples_collected < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, results = detector.find_hands(frame)
        landmarks = detector.get_landmarks(results)

        if landmarks:
            normalize=detector.normalize_landmarks(landmarks)
            save_sample(sign_label, normalize)
            samples_collected += 1

        cv2.putText(
            frame,
            f"{sign_label}: {samples_collected}/{SAMPLES_PER_SIGN}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        cv2.imshow("ASL Data Collector", frame)
        cv2.waitKey(1)

    print(f"Done! Collected {samples_collected} samples for {sign_label}")

def main():
    setup()
    detector=HandDetector()#creating a handdetector object 
    cap=cv2.VideoCapture(0)#creating a videocapture object, contains teh captured frame

    print("====ASL Data Collector===")
    print(f"Will collect {SAMPLES_PER_SIGN} samples per sign")

    while True:
        sign_label=input("\nEnter sign label to collect (or 'q' to quit):").upper()

        if sign_label=='Q':
            break

        collect_sign(sign_label, detector, cap)
        print(f"\nSamples saved to {DATA_FILE}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n Data collection complete!")
if __name__ == '__main__':
    main()


