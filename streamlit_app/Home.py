# load package
import os
# from cv2 import CAP_V4L2
import pickle

import cv2 as cv
import streamlit as st

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

## Create class for the videocapture
class VideoCamera:
    def __init__(self, cam_path):
        self.video = cv.VideoCapture(cam_path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        while True:
            success, image = self.video.read()

            # Recolor image to RGB
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            _, jpeg = cv.imencode('.jpg', image)
            return jpeg.tobytes()




st.set_page_config(page_title="Video Dashboard", page_icon="????",layout="wide")
st.title('Video Dashboard')
col1, col2 = st.columns(2)

with col1:
   st.header("Camera 1")
   FRAME_WINDOW1 = st.image([])

with col2:
   st.header("Camera 2")
   FRAME_WINDOW2 = st.image([])


col1, col2 = st.columns(2)

with col1:
   st.header("Camera 3")
   FRAME_WINDOW3 = st.image([])

with col2:
   st.header("Camera 4")
   FRAME_WINDOW4 = st.image([])





video_capture = cv.VideoCapture(0)
frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frame_width, frame_height])

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

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    FRAME_WINDOW1.image(rgb_frame)
    FRAME_WINDOW2.image(rgb_frame)
    FRAME_WINDOW3.image(rgb_frame)
    FRAME_WINDOW4.image(rgb_frame)

else:

    # conn.close()
    st.write('Stopped')