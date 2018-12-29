import EigenUtils
import cv2


class FaceRecognition:
    def __init__(self):
        self.eigen_face = EigenUtils.load_pickle('eigen_face.pickle')
        self.train_data = EigenUtils.load_pickle('train_vec.pickle')
        self.val_data = EigenUtils.load_pickle('va_vec.pickle')
        self.FACE_CASCADES = {
            'default': cv2.CascadeClassifier(
                'haarcascade_frontalface_default.xml'),
            'alt': cv2.CascadeClassifier(
                'haarcascade_frontalface_alt.xml'),
            'alt2': cv2.CascadeClassifier(
                'haarcascade_frontalface_alt2.xml'),
            'altree': cv2.CascadeClassifier(
                'haarcascade_frontalface_alt_tree.xml'),
            'profile': cv2.CascadeClassifier(
                'haarcascade_profileface.xml'),
            'LBP': cv2.CascadeClassifier(
                'lbpcascade_frontalface_improved.xml'),
            'cudaDef': cv2.CascadeClassifier(
                'haarcascade_frontalface_default_cuda.xml'
            )
        }

    def predict(self, image):
        if type(image) is None:
            return None, None
        else:
            crop, (x, y, w, h) = EigenUtils.get_cropped_face(image)
            if hasattr(crop, 'shape'):
                img_res = cv2.resize(crop, (32, 32))
                img_res = img_res.flatten()
                pred = EigenUtils.predict(img_res, self.eigen_face['average'], self.eigen_face['eigenface'],
                                          self.eigen_face['weight'],
                                          self.train_data['label'])
                cv2.rectangle(image, (x, y), (x + w, y + h), (200, 50, 25), 2)
                cv2.putText(image, '{}'.format(pred), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            return image, crop

    def predicts(self, image):
        crops = []
        self.last_predict = ''
        if not hasattr(image, 'shape'):
            return 0, 0
        else:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in self.FACE_CASCADES['default'].detectMultiScale(image=image_gray, scaleFactor=1.3,
                                                                               minNeighbors=5):
                crop = image_gray[y - 3:y + h + 3, x - 3:x + w + 3]

                if hasattr(crop, 'shape'):
                    img_res = cv2.resize(crop, (32, 32))
                    img_res = img_res.flatten()
                    pred = EigenUtils.predict(img_res, self.eigen_face['average'], self.eigen_face['eigenface'],
                                              self.eigen_face['weight'],
                                              self.train_data['label'])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (200, 50, 25), 2)
                    cv2.putText(image, '{}'.format(pred), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    cv2.putText(crop, '{}'.format(pred), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    if self.last_predict != pred:
                        crops.append(crop)
                        self.last_predict = pred
            return image, crops
