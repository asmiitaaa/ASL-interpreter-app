import cv2
import mediapipe as mp 
import numpy as np 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
#self refers to the current object of the hand detector class we are using
# we are adding all the attributes of that object inside the init constructor, like an opencv object

#EACH FRAME AND EACH RESULT CONTAINS ONLY ONE HAND 
class HandDetector:
    def __init__(self, model_path='hand_landmarker.task'):#model is the file that contains the trained neural network 
        #it takes the images and traces it to the coordinates, it is the model file containing the data, and we save the model file locally
        base_options=python.BaseOptions(model_asset_path=model_path)
        #this is an object that stores location of the model file, 
        options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7)#only one hand detected at a time, stores all the requirements/settings for the hand detector 
        self.detector=vision.HandLandmarker.create_from_options(options)#this is the detector object, we create the detecor object 

    def find_hands(self, frame):
        frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #opencv reads the images in bgr order, while mediapipe expects it in red green blue, so we swap those 
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )#we have to wrap the frame(pixel array-data frame^) in the mp.image object 
        results = self.detector.detect(mp_image)#e send the wrapped images to the detector 
        #media pipe processes the image and returns a results object 
        if results.hand_landmarks:#hand_landmarks is one of the fields the mediapipe puts into results 
            frame=self._draw_landmarks(frame, results)#if hand was found, draw the landmarks on the frame
        return frame, results 
    
    def _draw_landmarks(self, frame, results):
        for hand in results.hand_landmarks:
            h, w, _ = frame.shape

            points = []
            for landmark in hand:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]
            for start, end in connections:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        return frame

    def get_landmarks(self, results):#using the results, we get the landmarks 
        if not results.hand_landmarks:#if there are no handmarks found in the result
            return [] 
        hand=results.hand_landmarks[0]#there is only one, and we select that 
        landmarks=[[lm.x, lm.y, lm.z] for lm in hand]
        return landmarks

    def flatten_landmarks(self, landmarks):
        """
        Squashes [[x,y,z], [x,y,z], ...] into [x,y,z,x,y,z,...]
        63 numbers total — what the classifier will expect later.
        """
        return [coord for point in landmarks for coord in point]
        

