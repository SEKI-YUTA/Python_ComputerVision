import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('./video/video3.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(detection.location_data.relative_bounding_box)
            # mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            h,w, c = img.shape
            # print(w, h)
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (255, 0, 255),3)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20),cv2.FONT_HERSHEY_PLAIN,
                        3, (255,0,255),3)
            # cv2.circle(img, point1, 10, (0,0,255),10)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70,), cv2.FONT_HERSHEY_PLAIN,
    3, (180,255,0), 2)
    cv2.imshow('image', img)
    if cv2.waitKey(10) & 0xFF == 27:break