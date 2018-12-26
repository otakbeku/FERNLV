import cv2
import EigenUtils

class VideoCamera(object):
    def __init__(self, channel=0):
        self.video = cv2.VideoCapture(channel)


    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, image = self.video.read()

        return cv2.imencode('.jpg', image)[1].tobytes()
