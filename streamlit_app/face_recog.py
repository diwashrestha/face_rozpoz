import cv2 as cv
import numpy as np
import os
import pickle
from main_function import (encode_generator, face_compare, face_detect, face_encoding,
                           face_encoding_test, face_identify, face_loc, face_loc_test, face_mark)


# define the path of the detection and recognition model

# set face detection and recognition model
FACE_DETECTION_MODEL_PATH = "models/face_detection_yunet/face_detection_yunet_2022mar.onnx"
FACE_RECOGNITION_MODEL_PATH = "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Set face detection parameters
SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000
#tm = cv.TickMeter()


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



# Set up video capture from default webcam
video_capture = cv.VideoCapture(0)
frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frame_width, frame_height])


# Start video capture and face detection loop
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize the frame to match the size of the video capture
    frame = cv.resize(frame, (frame_width, frame_height))

    # Set the name of the recognized face to "Unknown" by default
    name = "Unknown"

    # Find all the faces and face encodings in the frame of video
    face_locations = face_detect(frame)


    # Test code
    if face_locations[1] is not None:

        print(face_locations[1][0])
        for idx, face in enumerate(face_locations[1]):
            print("Face {}".format(idx))
            # print("Face Location: {}".format(face))
            face_align = face_loc_test(frame, face)
            face_feature = face_encoding_test(face_align)
            print("Face Feature: {}".format(face_feature))
            name = face_identify(frame_face_feature=face_feature, known_face_encodings=known_face_encodings, known_face_names=known_face_names)
            face_mark(frame, face, name)

    # # Display the resulting image
    cv.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv.destroyAllWindows()
