import cv2
import os

from . import EigenUtils as eu


# TODO Need to be tested but for now everything is fine
class EigenFaceRecognition:
    """
    Class that provide to making a face recognition using Eigenface
    """

    def __init__(self, eigenface_path=None, train_path=None, val_path=None, face_cascade='default'):
        """
        Making a EigenFaceRecognition Instance

        :param eigenface_path: path of eigenface pickle
        :param train_path: path of train data pickle
        :param val_path: path of validation data pickle
        :param face_cascade: Face Cascade Classifier from OpenCV
        """
        if eigenface_path is None:
            self._eigen_face = eu.load_pickle('../Assets/eigen_face.pickle')
        else:
            self._eigen_face = eigenface_path
        if train_path is None:
            self._train_data = eu.load_pickle('../Assets/train_vec.pickle')
        else:
            self._train_data = train_path
        if val_path is None:
            self._val_data = eu.load_pickle('../Assets/val_vec.pickle')
        else:
            self._val_data = val_path
        self.face_cascade = eu.get_face_cascades(face_cascade)

    def set_eigen_face(self, eigenface_path=None):
        """

        :param eigenface_path: path of eigenface pickle
        :return:
        """
        if os.path.exists(eigenface_path):
            self._eigen_face = eigenface_path
        else:
            raise Exception(
                'File not found!'
            )

    def set_validation_data(self, val_path=None):
        """
        Set the validation data for instance

        :param val_path: path of validation data pickle
        :return:
        """
        if os.path.exists(val_path):
            self._val_data = val_path
        else:
            raise Exception(
                'File not found!'
            )

    def set_train_path(self, train_path=None):
        """
        Set the training data for instance

        :param train_path: path of training data pickle
        :return:
        """
        if os.path.exists(train_path):
            self._eigen_face = train_path
        else:
            raise Exception(
                'File not found!'
            )

    def set_face_cascade(self, face_cascade='default'):
        """
        Set the Face Cascade Classifier from OpenCV

        :param face_cascade: name of haarcascade that available from OpenCV which is:
            'default': default frontal face,
            'alt': alternative of frontal face,
            'alt2': another alternative for frontal face,
            'altree': another alternative version with tree ,
            'profile': Only detect the profile face,
            'LBP': XMl made with Local Binary Pattern,
             'cudaDef': A Cuda version of default
        :return:
        """
        self.face_cascade = eu.get_face_cascades(face_cascade)

    # TODO Not sure how this works fine
    def predict(self, image):
        """
        This method only give one predict from the given image
        :param image: an image. numpy ndarray
        :return: the image itself and the crop
        """
        if type(image) is None:
            return None, None
        else:
            crop, (x, y, w, h) = eu.get_cropped_face(image)
            if hasattr(crop, 'shape'):
                img_res = cv2.resize(crop, (32, 32))
                img_res = img_res.flatten()
                pred = eu.predict(img_res, self._eigen_face['average'], self._eigen_face['eigenface'],
                                  self._eigen_face['weight'],
                                  self._train_data['label'])
                cv2.rectangle(image, (x, y), (x + w, y + h), (200, 50, 25), 2)
                cv2.putText(image, '{}'.format(pred), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            return image, crop

    def predicts(self, image):
        """
        Predicts all the face available from the given image
        :param image: an image. numpy ndarray
        :return: an image and the list of the cropped face
        """
        crops = []
        self._last_predict = ''
        if not hasattr(image, 'shape'):
            return 0, 0
        else:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in self.face_cascade.detectMultiScale(image=image_gray, scaleFactor=1.3, minNeighbors=5):
                try:
                    crop = image_gray[y - 3:y + h + 3, x - 3:x + w + 3]
                except Exception as e:
                    print('Error: ', e)
                    crop = image_gray[y:y + h, x:x + w]

                if hasattr(crop, 'shape'):
                    img_res = cv2.resize(crop, (32, 32))
                    img_res = img_res.flatten()
                    pred = eu.predict(img_res, self._eigen_face['average'], self._eigen_face['eigenface'],
                                      self._eigen_face['weight'],
                                      self._train_data['label'])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (200, 50, 25), 2)
                    cv2.putText(image, '{}'.format(pred), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    if self._last_predict != pred:
                        crops.append(crop)
                        self._last_predict = pred
                    cv2.putText(crop, '{}'.format(self._last_predict), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 255), 2)
            return image, crops
