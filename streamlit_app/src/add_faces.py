import os
import shutil
import pickle
from main_function import (encode_generator, face_compare, face_detect, face_encoding,
                           face_encoding_test, face_identify, face_loc, face_loc_test, face_mark)

root_dir = "../Images/train"


def process_image(image_path, directory_name):
    # Create the directory
    os.makedirs(os.path.join(root_dir, directory_name), exist_ok=True)

    # Copy the image to the directory
    shutil.copy2(image_path, os.path.join(root_dir, directory_name))

    # Call encode creator function
    print("Face Encode generated")
    encode_generator(root_dir)
    encoding_file = open('../encode_file.p', 'rb')
    known_face_encode_name_list = pickle.load(encoding_file)
    encoding_file.close()
    print("Face Encoding created")


def another_function(directory_name):
    # This is where you would put the code for your other function
    print("Processing image in directory:", directory_name)


process_image("/home/diwashrestha/Projects/practo-deep-learno/open_cv_face/Images1/train/Karel Roden/karel2.jpeg",
              "karel roden")
