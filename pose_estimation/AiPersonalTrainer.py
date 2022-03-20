import cv2
import mediapipe as mp
import time
import PoseEstimationModule as ptm
import numpy as np

cap = cv2.VideoCapture('video/video4.mp4')
detector = ptm.PoseDetector()
pTime = 0
trainingCount = 0
dir = 0
while cap.isOpened():
    success, img = cap.read()
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #Right Arm
        # detector.findAngle(img, 12, 14, 16)
        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15)

        per = np.interp(angle, (200,330), (0, 100))
        print(per)
        if per == 100.0:
            if dir == 0:
                trainingCount+= 0.5
                dir = 1
        if per == 0.0:
            if dir == 1:
                trainingCount += 0.5
                dir = 0


    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime

    cv2.putText(img, f'Count: {trainingCount}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.putText(img, f'FPS: {fps}', (50,50),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),2)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:break