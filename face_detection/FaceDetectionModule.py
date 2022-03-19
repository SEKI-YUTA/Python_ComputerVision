import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)

    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(detection.location_data.relative_bounding_box)
                # mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                # print(w, h)
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                        int(bboxC.width * w), int(bboxC.height * h)
                bboxs.append([id, bbox, detection.score])
                # cv2.rectangle(img, bbox, (255, 0, 255),3)
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20),cv2.FONT_HERSHEY_PLAIN,
                                3, (255,0,255),3)
                # cv2.circle(img, point1, 10, (0,0,255),10)
            return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5,rt=2):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left x,y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right x,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left x,y
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        # Bottom Right x,y
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


def main():
    cap = cv2.VideoCapture('./video/video2.mp4')
    pTime = 0
    detector = FaceDetector()
    while cap.isOpened():
        success, img = cap.read()
        img, bboxs = detector.findFace(img,draw=False)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70,), cv2.FONT_HERSHEY_PLAIN,
                    3, (180, 255, 0), 2)
        cv2.imshow('image', img)
        if cv2.waitKey(10) & 0xFF == 27: break

if __name__ == "__main__":
    main()