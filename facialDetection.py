import cv2


class facialDetection:
    def __init__(self) -> None:
        pass
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    def faceDetector(self, gray_image, frame):
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        status = False
        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            if len(face) > 0:
                status = True
            else:
                status = False

        return frame, status

    def capture(self):
        cap = cv2.VideoCapture(1)
        ret, frame = cap.read()
        while cap.isOpened():
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output, status = self.faceDetector(gray, frame)
            cv2.imshow('Video', output)
            print(status)
            # return status
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # else:
            #     return status

        cap.release()
        cv2.destroyAllWindows()
