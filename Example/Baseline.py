'''
This file is used as a baseline of working face detection using Haar Cascade
filter. It used to make sure the face detection is correctly works. Later
this project will use the baseline's face detection before the face
recognition. Since the baseline works, I would continue the face recognition using some statistic model like eigenface, fisherface or ConvNet (using Keras layer).

'''

import cv2
from FERNLV import EigenUtils

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('../Assets/haarcascade_frontalface_default.xml')
eigen_face = EigenUtils.load_pickle('../Assets/eigen_face.pickle')
train_data = EigenUtils.load_pickle('../Assets/train_vec.pickle')
val_data = EigenUtils.load_pickle('../Assets/val_vec.pickle')

while True:
    if not cap.grab():
        break
    _, frame = cap.read()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_roi = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    for x, y, w, h in face_roi:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 50, 25), 2)
        crop = image_gray[y - 3:y + h - 3, x - 3:x + w - 3]
        crop = cv2.resize(crop, (32, 32))
        pred = EigenUtils.predict(crop.flatten(), eigen_face['average'], eigen_face['eigenface'],
                                  eigen_face['weight'],
                                  train_data['label'])
        cv2.putText(frame, '{}'.format(pred), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.imshow('Captured Face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
