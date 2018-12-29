from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea
from PyQt5.uic import loadUi
from PyQt5.Qt import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore, QtWidgets
import sys
import cv2

from face_recognition import FaceRecognition as fr
from CountPerSec import CountPerSec


def format_image(frame):
    image_format_qt = QImage.Format_Indexed8
    if len(frame.shape) == 3:
        if frame.shape[2] == 4:
            image_format_qt = QImage.Format_RGBA8888
        else:
            image_format_qt = QImage.Format_RGB888

    rgb_frame = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], image_format_qt)
    rgb_frame = rgb_frame.rgbSwapped()
    return rgb_frame


class FaceRecognitionWindow(QMainWindow):
    def __init__(self):
        super(FaceRecognitionWindow, self).__init__()
        loadUi('MainWindow.ui', self)
        self.frame = None
        self.StartButton.clicked.connect(self.start_webcam)
        self.StopButton.clicked.connect(self.stop_webcam)

        # THIS CODE ACCIDENTALLY WOKRS
        # ==========================================================
        # Widget -> Layout -> add widget -> layout -> scroll layout
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(710, 60, 370, 481))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.vertical_layout = QVBoxLayout(self)
        self.scroll_function = QScrollArea()
        self.scroll_function.setWidgetResizable(True)
        self.verticalLayout.addWidget(self.scroll_function)

        self.scroll_contents = QWidget()
        self.vertical_layout = QVBoxLayout(self.scroll_contents)
        self.scroll_function.setWidget(self.scroll_contents)
        # ==========================================================

        self.row_number = 0
        self.updated_face = 0
        self.blank = cv2.imread('blank.JPG')
        self.cps = CountPerSec().start()
        self.closed = False

        self.facereg = fr()
        print('Started')

    def __get_channel__(self):
        channel = self.channelText.toPlainText()
        if channel == '':
            print('Channel: 0')
            return 0
        else:
            print('Channel: ', channel)
            return int(channel)

    def start_webcam(self):
        print('start')
        self.cap = cv2.VideoCapture(self.__get_channel__())
        _, self.frame = self.cap.read()
        print('sekali')
        self.closed = False
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def __put_iterantion_text(self, frame, iteration_per_sec):
        cv2.putText(frame,
                    '{:.0f} iteration/sec'.format(iteration_per_sec),
                    (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255)
                    )
        return frame

    def update_frame(self):
        _, self.frame = self.cap.read()
        self.frame, crops = self.facereg.predicts(self.frame)
        if hasattr(crops, 'append') and len(crops) > 0:
            self.update_faces(crops)
        self.frame = self.__put_iterantion_text(self.frame, self.cps.count_per_sec())
        rgb_frame = format_image(self.frame)
        self.cps.increment()

        if hasattr(self.frame, 'shape'):
            self.display_image(rgb_frame)
        else:
            self.display_image(self.blank)

    def display_image(self, frame):
        self.ImageFrame.setPixmap(QPixmap.fromImage(frame))
        self.ImageFrame.setScaledContents(True)

    def stop_webcam(self):
        self.closed = True
        self.timer.stop()
        self.cap.release()
        print('stop')

    def update_faces(self, crops):
        for c in crops:
            label = QtWidgets.QLabel(self.scroll_contents)
            c_res = cv2.resize(c, (320, 240))
            c_res = format_image(c_res)
            label.setPixmap(QPixmap.fromImage(c_res))
            self.vertical_layout.addWidget(label)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionWindow()
    window.setWindowTitle('default')
    window.show()
    sys.exit(app.exec_())
