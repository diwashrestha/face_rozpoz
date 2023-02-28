import os
import pickle

import cv2 as cv
import numpy as np

# set face detection and recognition model
face_detection_model = "models/face_detection_yunet/face_detection_yunet_2022mar.onnx"
face_recognition_model = "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Set face detection parameters
SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000
tm = cv.TickMeter()

# Create face detector and recognizer objects
detector = cv.FaceDetectorYN.create(face_detection_model, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
recognizer = cv.FaceRecognizerSF.create(face_recognition_model, "")

# # Set face similarity thresholds
COSINE_SIMILARITY_THRESHOLD = 0.363
L2_SIMILARITY_THRESHOLD = 1.128


def encode_generator(root_dir):
    """
    Creates encoding for the images and saves it with labels for future use.
    :param root_dir: directory in which the image of known faces is
    """

    # Create an empty list to store the image file names and their respective directory names
    known_face_encode = []
    known_face_label = []

    # Loop through each actor directory
    for image_dir in os.listdir(root_dir):

        # Get the actor name from the directory name
        image_label = image_dir.lower()

        # Loop through each image file in the actor directory
        for filename in os.listdir(os.path.join(root_dir, image_dir)):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                # Open the image file using the image_read function
                img = image_read(os.path.join(root_dir, image_dir, filename))

                image_encode = face_encoding(img)
                if image_encode is not None:
                    known_face_encode.append(image_encode)
                    known_face_label.append(image_label)

    encode_known_list = [known_face_encode, known_face_label]
    file = open("encode_file.p", 'wb')
    pickle.dump(encode_known_list, file)
    file.close()


def face_encoding(img):
    """
    Creates the 128 dimension encoding of the face
    :param img: image path
    :return: encoding of the face
    """
    faces = face_detect(img)
    if faces[1] is not None:
        faces_align = face_loc(img, faces)
        faces_feature = recognizer.feature(faces_align)
        return faces_feature


def face_loc(img, faces):
    """
    Arrange the alignment of the face to make the face encoding easier
    :param img: image
    :param faces: face location in the image
    :return: aligned faces in the image
    """
    if faces[1] is not None:
        faces_align = recognizer.alignCrop(img, faces[1][0])
        return faces_align


def image_read(img_path):
    """
    Read the image from the image path and gives the read,resized image
    :param img_path: Image Path
    :return: Resized Image
    """
    image = cv.imread(cv.samples.findFile(img_path))
    image = cv.resize(image, (image.shape[1], image.shape[0]))
    return image


def face_detect(img):
    """
    Detect the face in the given frame or image
    :param img:
    :return: Detected face and its location using the numpy array
    """
    # img = image_read(img)
    tm.reset()
    tm.start()
    detector.setInputSize((img.shape[1], img.shape[0]))
    faces = detector.detect(img)
    tm.stop()
    return faces


def face_loc_test(frame, face):
    face_align = recognizer.alignCrop(frame, face)
    return face_align


def face_encoding_test(face_align):
    face_feature = recognizer.feature(face_align)
    return face_feature


def face_compare(known_face_encodings, face_encoding_to_check, cosine_similarity_threshold=0.363):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param cosine_similarity_threshold: How much distance between faces to consider it a match. higher value means higher similarity, max 1.0
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(recognizer.match(known_face_encodings, face_encoding_to_check) >= cosine_similarity_threshold)


def face_identify(frame_face_feature, known_face_encodings, known_face_names):
    """
    It will check if the encoding of the person in frame match with any previous face encoding.
    If it finds a match it gives the name of the person as output otherwise unknown.
    :param frame_face_feature:
    :param known_face_encodings:
    :param known_face_names:
    :return:
    """
    cosine_similarity_threshold = 0.363
    face_similarity = []

    for known_face in known_face_encodings:
        print(f"known_face:{known_face}")
        print(f"frame_face_feature: {frame_face_feature}")
        cosine_score = recognizer.match(frame_face_feature, known_face, cv.FaceRecognizerSF_FR_COSINE)
        face_similarity.append(cosine_score)

    print(f"Face Similarity: {face_similarity}")
    face_match = [score > cosine_similarity_threshold for score in face_similarity]
    print(f"Face Match: {face_match}")

    if any(face_match):
        best_match_index = np.argmax(face_similarity)
        face_identity = known_face_names[best_match_index]
    else:
        face_identity = "Unknown"

    return face_identity


def face_mark(input, face_location, face_name):
    """
    Creates the box around the face along with there name.
    :param input: Video frame
    :param face_location: Location of the detected face in the frame to make the box
    :param face_name: Name of the detected/identified people
    """
    thickness = 2
    coords = face_location[:-1].astype(np.int32)
    cv.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 1)
    # cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
    # cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
    # cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
    # cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
    # cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, face_name, (coords[0] + 6, coords[1] + coords[3] + 16), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0),
               1)
    cv.putText(input, 'FPS: {:.2f}'.format(tm.getFPS()), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    print(coords)
