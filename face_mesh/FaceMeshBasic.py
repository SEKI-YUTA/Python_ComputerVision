import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('./video/video1.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec,drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                h, w, c = img.shape
                x, y, = int(lm.x * w), int(lm.y * h)
                print(id, x, y)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img,f'FPS:{int(fps)}',(40, 60),cv2.FONT_HERSHEY_PLAIN,3,(255, 0, 255),3)
    cv2.imshow('image', img)



    if cv2.waitKey(1) & 0xFF == 27:break