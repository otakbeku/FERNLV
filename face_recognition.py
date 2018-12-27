import EigenUtils
import cv2


class FaceRecognition:
    def __init__(self):
        self.eigen_face = EigenUtils.load_pickle('eigen_face.pickle')
        self.train_data = EigenUtils.load_pickle('train_vec.pickle')
        self.val_data = EigenUtils.load_pickle('va_vec.pickle')

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
        if not hasattr(image, 'shape'):
            # print('sini')
            return 0, 0
        else:
            for crop, (x, y, w, h) in EigenUtils.get_cropped_faces(image):
                if hasattr(crop, 'shape'):
                    img_res = cv2.resize(crop, (32, 32))
                    img_res = img_res.flatten()
                    pred = EigenUtils.predict(img_res, self.eigen_face['average'], self.eigen_face['eigenface'],
                                              self.eigen_face['weight'],
                                              self.train_data['label'])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (200, 50, 25), 2)
                    cv2.putText(image, '{}'.format(pred), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    cv2.putText(crop, '{}'.format(pred), (40, 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    crops.append(crop)
            return image, crops
