import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexoty=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexoty = modelComplexoty
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        model_complexity=self.modelComplexoty, min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        # static_image_mode = False,
        # max_num_hands = 2,
        # model_complexity = 1,
        # min_detection_confidence = 0.5,
        # min_tracking_confidence = 0.5

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            # 手の配列
            # print(results.multi_hand_landmarks)
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx,cy)
                self.lmList.append([id, cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self,):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return  fingers
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # results = hands.process(imgRGB)
    # if results.multi_hand_landmarks:
    #     # 手の配列
    #     # print(results.multi_hand_landmarks)
    #     for handLms in results.multi_hand_landmarks:
    #         # ポイントの配列
    #         for id, lm in enumerate(handLms.landmark):
    #             # print(id, lm)
    #             h,w,c = img.shape
    #             cx,cy = int(lm.x*w), int(lm.y*h)
    #             print(id, cx,cy)
    #             if id==0:
    #                 cv2.circle(img, (cx,cy), 25, (255, 0, 255), cv2.FILLED)
    #         mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while (cap.isOpened()):
        success, img = cap.read()
        img = detector.findHands(img=img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0 , 255), 3)
        cv2.imshow('Image', img)
        
        if cv2.waitKey(1) & 0xFF == 27: break
    
    
if __name__ == '__main__':
    main()