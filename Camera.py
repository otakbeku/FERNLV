import cv2
from face_recognition import FaceRecognition as fr


class VideoCamera(object):
    def __init__(self, channel=0):
        self.video = cv2.VideoCapture(channel)
        self.facereg = fr()
        self.blank = cv2.imread('blank.JPG')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, image = self.video.read()
        if ret:
            image, crop = self.facereg.predict(image)
            if not hasattr(image, 'shape') and not hasattr(crop, 'shape'):
                return cv2.imencode('.jpg', self.blank)[1].tobytes(), 0
            elif not hasattr(crop, 'shape') and hasattr(image, 'shape'):
                return cv2.imencode('.jpg', image)[1].tobytes(), 0
            elif hasattr(crop, 'shape') and hasattr(image, 'shape'):
                return cv2.imencode('.jpg', image)[1].tobytes(), cv2.imencode('.jpg', crop)[1].tobytes()
        else:
            return cv2.imencode('.jpg', self.blank)[1].tobytes(), 0

    def get_frames(self):
        _, image = self.video.read()
        print(self.video.isOpened())
        image, crop = self.facereg.predicts(image)
        if not hasattr(image, 'shape') and not hasattr(crop, 'append'):
            return cv2.imencode('.jpg', self.blank)[1].tobytes(), 0
        elif not hasattr(crop, 'append') and hasattr(image, 'shape'):
            return cv2.imencode('.jpg', image)[1].tobytes(), 0
        elif hasattr(crop, 'append') and hasattr(image, 'shape'):
            return cv2.imencode('.jpg', image)[1].tobytes(), crop
