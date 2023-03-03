from threading import Thread
import cv2
import time


class vStream:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            _, self.frame = self.capture.read()

    def getFrame(self):
        return self.frame


cam1 = vStream(-1)
cam2 = vStream(2)
while True:
    try:
        myFrame1 = cam1.getFrame()
        myFrame2 = cam2.getFrame()
        cv2.imshow('1cam', myFrame1)
        cv2.imshow('2cam', myFrame2)
    except:
        print('frame not available')
        if cv2.waitKey(1) == ord('q'):
            cam1.capture.release()
            cam2.capture.release()
            cv2.destroyAllWindows()
            exit("l")
            break
