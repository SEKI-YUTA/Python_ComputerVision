import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon = 0.5, minTrakingCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrakingCon = minTrakingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=self.maxFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    h, w, c = img.shape
                    x, y, = int(lm.x * w), int(lm.y * h)
                    if id == 1:
                        # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                        cv2.circle(img,(x,y),10,(0,0,255),3,cv2.FILLED)
                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces

#
#
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = faceMesh.process(imgRGB)
#     if results.multi_face_landmarks:
#         for faceLms in results.multi_face_landmarks:
#             mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec,drawSpec)
#             for id, lm in enumerate(faceLms.landmark):
#                 # print(lm)
#                 h, w, c = img.shape
#                 x, y, = int(lm.x * w), int(lm.y * h)
#                 print(id, x, y)







def main():
    cap = cv2.VideoCapture('./video/video3.mp4')
    pTime = 0
    detector = FaceMeshDetector()
    while cap.isOpened():
        success, img = cap.read()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        img,faces = detector.findFaceMesh(img)
        if len(faces)!= 0:
            print(len(faces))
        cv2.putText(img, f'FPS:{int(fps)}', (40, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == 27: break

if __name__ == "__main__":
    main()