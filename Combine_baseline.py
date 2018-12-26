'''
This file is used as a baseline of working face detection using Haar Cascade
filter. It used to make sure the face detection is correctly works. Later
this project will use the baseline's face detection before the face
recognition. Since the baseline works, I would continue the face recognition using some statistic model like eigenface, fisherface or ConvNet (using Keras layer).

Baseline:
1. Using Haar Cascade
2. Using Dlib -> face
'''

import dlib
import cv2
import numpy as np


def shape_to_np(shape, dtype='int'):
    print(type(shape))
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def rect_to_roi(rect: dlib.rectangle):
    x, y = rect.tl_corner().x, rect.tl_corner().y
    w, h = rect.br_corner().x, rect.br_corner().y
    return x, y, w, h


# cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture('F:\FSR\FERNLV\gtkom.mp4')
cap = cv2.VideoCapture('F:\FSR\FERNLV\AFW1.mp4')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

p = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

while True:
    if not cap.grab():
        break
    _, frame = cap.read()
    face_roi = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 2, 3)

    for x, y, w, h in face_roi:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 50, 25), 2)
        cropped = frame[y:y + h, x:x + w]
        # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        face_roi = detector(rgb_frame, 0)

        for i, rect in enumerate(face_roi):
            shape = predictor(rgb_frame, rect)
            shape = shape_to_np(shape)
            xi, yi, wi, hi = rect_to_roi(rect)
            cv2.rectangle(frame, (x + xi, y + yi), (x + wi, y + hi), (200, 50, 25), 2)

            for (xc, yc) in shape:
                cv2.circle(frame, (xc + x, yc  + y), 2, (0, 255, 0), -1)

    cv2.imshow('Captured Face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
