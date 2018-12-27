# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1333, 677)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.StartButton = QtWidgets.QPushButton(self.centralwidget)
        self.StartButton.setGeometry(QtCore.QRect(30, 580, 101, 51))
        self.StartButton.setObjectName("StartButton")
        self.StopButton = QtWidgets.QPushButton(self.centralwidget)
        self.StopButton.setGeometry(QtCore.QRect(160, 580, 101, 51))
        self.StopButton.setObjectName("StopButton")
        self.ImageFrame = QtWidgets.QLabel(self.centralwidget)
        self.ImageFrame.setGeometry(QtCore.QRect(20, 60, 640, 480))
        self.ImageFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.ImageFrame.setText("")
        self.ImageFrame.setObjectName("ImageFrame")
        self.channelText = QtWidgets.QTextEdit(self.centralwidget)
        self.channelText.setGeometry(QtCore.QRect(370, 580, 111, 31))
        self.channelText.setObjectName("channelText")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(270, 590, 91, 21))
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(190, 20, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(710, 30, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.ClearButton = QtWidgets.QPushButton(self.centralwidget)
        self.ClearButton.setGeometry(QtCore.QRect(710, 580, 101, 51))
        self.ClearButton.setObjectName("ClearButton")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(730, 90, 256, 192))
        self.tableWidget.setRowCount(3)
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1333, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face Recognition Using Eigenface"))
        self.StartButton.setText(_translate("MainWindow", "Start"))
        self.StopButton.setText(_translate("MainWindow", "Stop"))
        self.label_3.setText(_translate("MainWindow", "Channel webcam"))
        self.label_5.setText(_translate("MainWindow", "Live Video"))
        self.label_7.setText(_translate("MainWindow", "List of Captured Face"))
        self.ClearButton.setText(_translate("MainWindow", "Clear"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "x"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "y"))

