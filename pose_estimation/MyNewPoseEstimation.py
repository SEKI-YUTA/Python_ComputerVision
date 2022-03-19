import cv2
import mediapipe
import time
import PoseEstimationModule

cap = cv2.VideoCapture('./video/video1.mp4')
pTime = 0
detector = PoseEstimationModule.poseDetector()
while cap.isOpened():
    success, img = cap.read()
    img = detector.findPose(img=img)
    lmList = detector.findPosition(img=img)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 255, 0))
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime


    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == 27: break