import cv2

from .FaceRecognition import EigenFaceRecognition as fr


# TODO Fixing the VideoCamera for multipurpose ...
class VideoCamera(object):
    """
    Making my own frame show with threading by various tutorial site. This not work as I want but still works
    """
    def __init__(self, channel=0):
        """
        Initialize the object
        :param channel: desire channel for webcam. 0 for default webcam
        """
        self.video = cv2.VideoCapture(channel)
        self.face_reg = fr()
        self.blank = cv2.imread('.\Assests\\blank.JPG')

    def __del__(self):
        """
        Release the capture
        :return:
        """
        self.video.release()

# TODO Doesn't works
    def get_frame(self):
        """
        Getting the frame and predict the face
        :return:
        """
        ret, image = self.video.read()
        if ret:
            image, crop = self.face_reg.predict(image)
            if not hasattr(image, 'shape') and not hasattr(crop, 'shape'):
                return cv2.imencode('.jpg', self.blank)[1].tobytes(), 0
            elif not hasattr(crop, 'shape') and hasattr(image, 'shape'):
                return cv2.imencode('.jpg', image)[1].tobytes(), 0
            elif hasattr(crop, 'shape') and hasattr(image, 'shape'):
                return cv2.imencode('.jpg', image)[1].tobytes(), cv2.imencode('.jpg', crop)[1].tobytes()
        else:
            return cv2.imencode('.jpg', self.blank)[1].tobytes(), 0

    # TODO Doesn't works too
    def get_frame_ver2(self):
        """
        Getting the frame and predict the face. Version 2
        :return:
        """
        _, image = self.video.read()
        image, crop = self.face_reg.predicts(image)
        if not hasattr(image, 'shape') and not hasattr(crop, 'append'):
            return cv2.imencode('.jpg', self.blank)[1].tobytes(), 0
        elif not hasattr(crop, 'append') and hasattr(image, 'shape'):
            return cv2.imencode('.jpg', image)[1].tobytes(), 0
        elif hasattr(crop, 'append') and hasattr(image, 'shape'):
            return cv2.imencode('.jpg', image)[1].tobytes(), crop
