import EigenUtils
import cv2


class FaceRecognition:
    def __init__(self):
        self.eigen_face = EigenUtils.load_pickle('eigen_face.pickle')
        self.train_data = EigenUtils.load_pickle('train_vec.pickle')
        self.val_data = EigenUtils.load_pickle('va_vec.pickle')

    def predict(self, image):
        # Tambah fungsi buat deteksi wajah
        img_res = cv2.resize(image, (32, 32))
        img_res = img_res.flatten()
        pred = EigenUtils.predict(img_res, self.eigen_face['average'], self.eigen_face['eigenface'], self.eigen_face['weight'],
                           self.train_data['label'])

