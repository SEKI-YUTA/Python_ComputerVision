import cv2

class FaceDetector:
    def __init__(self):
        self.facecCascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_alt.xml')
    def findFace(self,img):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.facecCascade.detectMultiScale(grey)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
        return img
facecCascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)


def main():
    detector = FaceDetector()
    while cap.isOpened():
        success, img = cap.read()
        img = detector.findFace(img)
        cv2.imshow('image', img)
        cv2.CascadeClassifier()
        if cv2.waitKey(1) & 0xFF == 27: break

if __name__ == "__main__":
    main()