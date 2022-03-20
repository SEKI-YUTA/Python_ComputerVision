import cv2
import mediapipe as mp
import time
import os
import MyHandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetector()
tipIds = [4, 8, 12, 16, 20]

pTime = 0
exImg = cv2.imread('exImage.jpg')
exImgShape = exImg.shape
# print(exImgShape)
while cap.isOpened():
    success, img = cap.read()
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime
    # img[0:exImgShape[0],0:exImgShape[1]] = exImg
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # 4,8,12,16,20
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        totalFingers = fingers.count(1)
        cv2.putText(img, f'Count:{totalFingers}', (300, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.putText(img, f'FPS:{fps}', (50, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:break