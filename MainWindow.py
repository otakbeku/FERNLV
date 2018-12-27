from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.uic import loadUi
from PyQt5.Qt import QImage, QPixmap, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore, QtWidgets
import sys
import cv2
from threading import Thread

from face_recognition import FaceRecognition as fr


class FaceRecognitionWindow(QMainWindow):
    def __init__(self):
        super(FaceRecognitionWindow, self).__init__()
        loadUi('MainWindow.ui', self)
        self.frame = None
        self.StartButton.clicked.connect(self.start_webcam)
        self.StopButton.clicked.connect(self.stop_webcam)
        self.ClearButton.clicked.connect(self.clear_detected_face)

        self.detected_faces = []
        self.tableWidget = QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(710, 60, 581, 481))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.setRowCount(0)
        self.row_number = 0
        self.updated_face = 0
        self.blank = cv2.imread('blank.JPG')

        # self.facereg = fr()

    def __get_channel__(self):
        channel = self.channelText.toPlainText()
        if channel == '':
            print('Channel: 0')
            return 0
        else:
            print('Channel: ', channel)
            return int(channel)

    @staticmethod
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

    def start_webcam(self):
        print('start')
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        _, self.frame = self.cap.read()
        # self.frame, crops = self.facereg.predicts(self.frame)
        # if hasattr(crops, 'append'):
        #     self.updated_face = len(crops)
        #     [self.detected_faces.append(format_image(cv2.resize(c, (50, 50)))) for c in crops]
        if hasattr(self.frame, 'shape'):
            self.display_image(self.frame)
        else:
            self.display_image(self.blank)

    def update_detected_face(self):
        for i in range(self.updated_face):
            self.row_number += 1
            self.tableWidget.setItem(self.row_number, 0, self.detected_faces[i + self.row_number - 1])

    def display_image(self, frame):
        rgb_frame = self.format_image(frame)
        self.ImageFrame.setPixmap(QPixmap.fromImage(rgb_frame))
        self.ImageFrame.setScaledContents(True)

    def stop_webcam(self):
        self.timer.stop()
        self.cap.release()
        print('stop')

    def clear_detected_face(self):
        self.detected_faces.clear()
        self.row_number = 0
        self.tableWidget.setRowCount(self.row_number)


class TableWidget(QWidget):
    def __init__(self):
        super(TableWidget, self).__init__()
        self.table()

    def table(self):
        self.table_widget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tablewidget)
        self.setLayout(self.layout)
        self.show()

    def table_widget(self):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(1)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setItem(0, 0, QTableWidgetItem("Hello"))
        self.tableWidget.move(300, 300)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionWindow()
    window.setWindowTitle('default')
    window.show()
    sys.exit(app.exec_())
