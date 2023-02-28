import cv2
import face_recognition
import pickle
import os


class ImageEncoder:
    def __init__(self, cst_folder_path, off_folder_path):
        self.cst_folder_path = cst_folder_path
        self.off_folder_path = off_folder_path
        self.img_list = []
        self.people_ids = []

    def import_images(self):
        cst_path_list = os.listdir(self.cst_folder_path)
        off_path_list = os.listdir(self.off_folder_path)
        for cst_path in cst_path_list:
            self.img_list.append(cv2.imread(os.path.join(self.cst_folder_path, cst_path)))
            self.people_ids.append(os.path.splitext(cst_path)[0])
        for off_path in off_path_list:
            self.img_list.append(cv2.imread(os.path.join(self.off_folder_path, off_path)))
            self.people_ids.append(os.path.splitext(off_path)[0])

    def find_encodings(self, images_list):
        encode_list = []
        for img in images_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encode_list.append(encode)
        return encode_list

    def save_encodings(self):
        self.import_images()
        encode_list_known = self.find_encodings(self.img_list)
        encode_list_known_with_ids = [encode_list_known, self.people_ids]
        file = open("encode_file.p", 'wb')
        pickle.dump(encode_list_known_with_ids, file)
        file.close()


if __name__ == '__main__':
    encoder = ImageEncoder('Images/customer', 'Images/offender')
    encoder.save_encodings()
