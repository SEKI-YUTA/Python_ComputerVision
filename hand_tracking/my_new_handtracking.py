import time
import cv2
from matplotlib.pyplot import draw
import mediapipe as mp
import MyHandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
while (cap.isOpened()):
    success, img = cap.read()
    img = detector.findHands(img=img)
    lmList = detector.findPosition(img,draw=True)

    if len(lmList) != 0:
        print(lmList[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == 27: break