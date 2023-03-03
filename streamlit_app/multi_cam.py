from threading import Thread
import time
import os
import pickle
import cv2 as cv


#from Home import VideoCamera
from src.main_function import (encode_generator, face_detect, face_encoding_test, face_identify, face_loc_test,
                               face_mark)

# set face detection and recognition model
FACE_DETECTION_MODEL_PATH = "models/face_detection_yunet/face_detection_yunet_2022mar.onnx"
FACE_RECOGNITION_MODEL_PATH = "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Set face detection parameters
SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000
tm = cv.TickMeter()


# Create face detector and recognizer objects
detector = cv.FaceDetectorYN.create(FACE_DETECTION_MODEL_PATH, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
recognizer = cv.FaceRecognizerSF.create(FACE_RECOGNITION_MODEL_PATH, "")

# Set face similarity thresholds
COSINE_SIMILARITY_THRESHOLD = 0.363
L2_SIMILARITY_THRESHOLD = 1.128

# Check if the encoding file exists
if os.path.exists("encode_file.p"):
    # If the file exists, load its contents and store them in variables
    encoding_file = open('encode_file.p', 'rb')
    known_face_encode_name_list = pickle.load(encoding_file)
    encoding_file.close()
    known_face_encodings, known_face_names = known_face_encode_name_list
else:
    # If the file does not exist, generate the face encodings and store their output in variables
    encode_generator("Images/train")
    encoding_file = open('encode_file.p', 'rb')
    known_face_encode_name_list = pickle.load(encoding_file)
    encoding_file.close()
    known_face_encodings, known_face_names = known_face_encode_name_list

class vStream:
    def __init__(self, src):
        self.capture = cv.VideoCapture(src)
        self.detector = cv.FaceDetectorYN.create(FACE_DETECTION_MODEL_PATH, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
        self.frame_width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        #self.detector.setInputSize([self.frame_width, self.frame])
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            _, self.frame = self.capture.read()

    def getFrame(self):
        # Resize the frame to match the size of the video capture
        self.frame = cv.resize(self.frame, (self.frame_width, self.frame_height))
        return self.frame


cam1 = vStream(-1)
cam2 = vStream("rtsp://admin:admin@192.168.1.110") 
cam3 = vStream(2)
while True:
    try:
        myFrame1 = cam1.getFrame()
        myFrame2 = cam2.getFrame()
        myFrame3 = cam3.getFrame()
        
        # face recognition in camera 1
        face_locations1 = face_detect(myFrame1)
        
            # Test code
        if face_locations1[1] is not None:

            print(face_locations1[1][0])
            for idx, face in enumerate(face_locations1[1]):
                print("Face {}".format(idx))
                # print("Face Location: {}".format(face))
                face_align1 = face_loc_test(myFrame1, face)
                face_feature1 = face_encoding_test(face_align1)
                print("Face Feature: {}".format(face_feature1))
                name = face_identify(frame_face_feature=face_feature1, known_face_encodings=known_face_encodings, known_face_names=known_face_names)
                face_mark(myFrame1, face, name)        
        
        # face recognition in camera 2
        face_locations2 = face_detect(myFrame2)
        
            # Test code
        if face_locations2[1] is not None:

            print(face_locations2[1][0])
            for idx, face in enumerate(face_locations2[1]):
                print("Face {}".format(idx))
                # print("Face Location: {}".format(face))
                face_align2 = face_loc_test(myFrame2, face)
                face_feature2 = face_encoding_test(face_align2)
                print("Face Feature: {}".format(face_feature2))
                name = face_identify(frame_face_feature=face_feature2, known_face_encodings=known_face_encodings, known_face_names=known_face_names)
                face_mark(myFrame2, face, name)        
        
        
        # face recognition in camera 3
        face_locations3 = face_detect(myFrame3)
        
            # Test code
        if face_locations3[1] is not None:

            print(face_locations3[1][0])
            for idx, face in enumerate(face_locations3[1]):
                print("Face {}".format(idx))
                # print("Face Location: {}".format(face))
                face_align3 = face_loc_test(myFrame3, face)
                face_feature3 = face_encoding_test(face_align3)
                print("Face Feature: {}".format(face_feature3))
                name = face_identify(frame_face_feature=face_feature3, known_face_encodings=known_face_encodings, known_face_names=known_face_names)
                face_mark(myFrame3, face, name)
        
        cv.imshow('1cam', myFrame1)
        cv.imshow('2cam', myFrame2)
        cv.imshow('3cam', myFrame3)
    except:
        print('frame not available')
    if cv.waitKey(1) == ord('q'):
        cam1.capture.release()
        cam2.capture.release()
        cam3.capture.release()
        cv.destroyAllWindows()
        exit("l")
        break
