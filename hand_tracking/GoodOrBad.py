import time
import cv2
import mediapipe as mp
import MyHandTrackingModule as htm

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while cap.isOpened():
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # display good or bad action
        # id 4 id 20
        print(lmList[4], lmList[20])
        x1, y1 = lmList[4][1],lmList[4][2]
        x2, y2 = lmList[20][1], lmList[20][2]

        if lmList[4][2] + 100 < lmList[20][2]:
            cv2.putText(img, 'Good!', (x1,y1), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        elif lmList[4][2] > lmList[20][2] + 100:
            cv2.putText(img, 'Bad', (x2,y2), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF == 27:break