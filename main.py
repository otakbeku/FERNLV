import EigenUtils
import cv2

train_data, train_labels, val_data, val_labels, _, _ = EigenUtils.split_data_train_test_val('dataset', base_dir='base_dir')

eigen_face, weight, average = EigenUtils.create_eigen_face_vector(train_data)

test = 'F:\FSR\FERNLV\dataset\Thor\\1.JPG'
img = cv2.imread(test, 0)
img_res = cv2.resize(img, (32, 32))
img_res = img_res.flatten()

print(EigenUtils.predict(img_res, average, eigen_face, weight, train_labels))
# Uji Performa
eigen_face = EigenUtils.load_pickle('eigen_face.pickle')
train_data = EigenUtils.load_pickle('train_vec.pickle')
val_data = EigenUtils.load_pickle('va_vec.pickle')
EigenUtils.performance_measure(eigen_face, train_labels=train_data['label'], val_data=val_data['vec'],
                               val_label=val_data['label'])
