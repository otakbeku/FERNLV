import numpy as np
import cv2
import os
from tensorflow.python.platform import gfile
import re
import shutil
import hashlib
from tensorflow.python.util import compat
import pickle

EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
MAX_NUM_IMAGES_PER_CLASS = 100

FACE_CASCADES = {
    'default': cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml'),
    'alt': cv2.CascadeClassifier(
        'haarcascade_frontalface_alt.xml'),
    'alt2': cv2.CascadeClassifier(
        'haarcascade_frontalface_alt2.xml'),
    'altree': cv2.CascadeClassifier(
        'haarcascade_frontalface_alt_tree.xml'),
    'profile': cv2.CascadeClassifier(
        'haarcascade_profileface.xml'),
    'LBP': cv2.CascadeClassifier(
        'lbpcascade_frontalface_improved.xml'),
    'cudaDef': cv2.CascadeClassifier(
        'haarcascade_frontalface_default_cuda.xml'
    )
}

CASCADE_DEFAULT = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def vec_normalization(eigen_vector):
    eigen_vector = np.asarray(eigen_vector)
    max_pixel = eigen_vector.max()
    min_pixel = eigen_vector.min()
    for index in range(len(eigen_vector)):
        new_pixel = []
        for indexPixel in range(len(eigen_vector[index])):
            value = 255 * (eigen_vector[index, indexPixel] - min_pixel) / (max_pixel - min_pixel)

            value = np.uint8(value.real)
            new_pixel.append(value)
        eigen_vector[index] = new_pixel
    return eigen_vector


def get_cropped_face(image,
                     y_plus=0, y_min=0, x_min=0, x_plus=0,
                     face_cascade: cv2.CascadeClassifier =
                     FACE_CASCADES['default']):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # d = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    d = CASCADE_DEFAULT.detectMultiScale(image_gray, 1.3, 5)
    try:
        [[x, y, w, h]] = d
        crop = image_gray[y - y_min:y + h + y_plus, x - x_min:x + w + x_plus]
        return crop, (x, y, w, h)

    except Exception as e:
        print("Err: ", str(e))
        return None, (0, 0, 0, 0)


def get_cropped_faces(image,
                      y_plus=0, y_min=0, x_min=0, x_plus=0,
                      face_cascade: cv2.CascadeClassifier =
                      FACE_CASCADES['default']):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # d = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    d = CASCADE_DEFAULT.detectMultiScale(image_gray, 1.3, 5)
    try:
        for x, y, w, h in d:
            crop = image_gray[y - y_min:y + h + y_plus, x - x_min:x + w + x_plus]
            return crop, (x, y, w, h)

    except Exception as e:
        print("Err: ", str(e))
        return None, (0, 0, 0, 0)


def image_to_vec(image, size):
    temp: np.ndarray = cv2.resize(image, (size, size))
    vec = temp.flatten()
    return vec


def split_data_train_test_val(image_dir: str, base_dir: str,
                              testing_percentage=0,
                              validation_percentage=10,
                              size=32, save=True,
                              cropped=False):
    moves = 'Moves {} to {}'
    val_data = []
    val_labels = []
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        os.mkdir(base_dir)
    else:
        os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train_dir')
    val_dir = os.path.join(base_dir, 'val_dir')
    test_dir = os.path.join(base_dir, 'test_dir')
    print('Create train dir')
    os.mkdir(train_dir)
    print('Create val dir')
    os.mkdir(val_dir)
    print('Create test dir')
    os.mkdir(test_dir)

    sub_dirs = [os.path.join(image_dir, item) for item in os.listdir(image_dir)]
    sub_dirs = sorted(item for item in sub_dirs if os.path.isdir(item))
    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print('Looking for images in \'' + sub_dir + '\'')

        for ext in EXTENSIONS:
            file_glob = os.path.join(image_dir, dir_name, '*.' + ext)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print('No Files Found')
            continue
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        for file_name in file_list:
            val_sub_dir = os.path.join(val_dir, dir_name)
            if not os.path.exists(val_sub_dir):
                print('Create ', val_sub_dir)
                os.mkdir(val_sub_dir)

            train_sub_dir = os.path.join(train_dir, dir_name)
            if not os.path.exists(train_sub_dir):
                print('Create ', train_sub_dir)
                os.mkdir(train_sub_dir)

            test_sub_dir = os.path.join(test_dir, dir_name)
            if not os.path.exists(test_sub_dir):
                print('Create ', test_sub_dir)
                os.mkdir(test_sub_dir)

            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            temp: np.ndarray = cv2.imread(file_name, 0)
            if cropped:
                temp, shape = get_cropped_face(temp)
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                if os.path.exists(os.path.join(val_sub_dir, base_name)):
                    continue
                shutil.copy(file_name, val_sub_dir)
                print(moves.format(base_name, val_sub_dir))
                val_data.append(image_to_vec(temp, size))
                val_labels.append(label_name)

            if percentage_hash < (testing_percentage + validation_percentage) and testing_percentage > 0:
                if os.path.exists(os.path.join(test_sub_dir, base_name)):
                    continue
                shutil.copy(file_name, test_sub_dir)
                print(moves.format(base_name, test_sub_dir))
                test_data.append(image_to_vec(temp, size))
                test_labels.append(label_name)

            else:
                if os.path.exists(os.path.join(train_sub_dir, base_name)):
                    continue
                shutil.copy(file_name, train_sub_dir)
                print(moves.format(base_name, train_sub_dir))
                train_data.append(image_to_vec(temp, size))
                train_labels.append(label_name)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    val_data = np.array(val_data)
    train_dict = {'vecs': train_data, 'labels': train_labels}
    test_dict = {'vecs': test_data, 'labels': test_labels}
    val_dict = {'vecs': val_data, 'labels': val_labels}
    if save:
        with open('train_vec.pickle', 'wb') as f:
            pickle.dump(train_dict, f)
        with open('val_vec.pickle', 'wb') as f:
            pickle.dump(val_dict, f)
        if testing_percentage > 0:
            with open('test_vec.pickle', 'wb') as f:
                pickle.dump(test_dict, f)
    print('Done')
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def get_eigen_weight(eigen_face, subtracted_average):
    weight = []
    for index in range(len(subtracted_average[0])):
        weight_temp = []
        for value in eigen_face:
            temp = np.transpose(np.reshape(value, (-1, 1)))
            try:
                weight_temp.append(
                    np.dot(temp, np.transpose(subtracted_average)[index])
                )
            except Exception as e:
                weight_temp.append(
                    np.dot(temp, subtracted_average[index])
                )
        weight.append(weight_temp)
    return weight


def create_eigen_face_vector(data: np.ndarray, normalization=True, max_component=25, save=True):
    average = data.mean()
    subtracted_avg = data - average
    subtracted_avg = subtracted_avg.T
    x, y = subtracted_avg.shape
    if y > x:
        mat_covariance = np.dot(subtracted_avg, subtracted_avg.T)
    else:
        mat_covariance = np.dot(subtracted_avg.T, subtracted_avg)

    U, s, V = np.linalg.svd(mat_covariance, full_matrices=True, compute_uv=True)
    V = np.transpose(V)
    print('U: {}\nV: {}\ns: {}'.format(U.shape, V.shape, s.shape))
    eigen_value_temp = s ** -0.5
    eigen_vector_u = []
    for i, value in enumerate(V):
        try:
            temp = np.dot(subtracted_avg, value)
        except Exception as e:
            temp = np.dot(subtracted_avg.T, value)
        eigen_vector_u.append(temp / eigen_value_temp[i])

    if normalization:
        eigen_vector_u = vec_normalization(eigen_vector_u)

    eigen_face = eigen_vector_u[:max_component]
    weight = get_eigen_weight(eigen_face, subtracted_avg)

    data = {'eigenface': eigen_face, 'weight': weight, 'average': average}
    if save:
        with open('eigen_face.pickle', 'wb') as f:
            pickle.dump(data, f)

    return eigen_face, weight, average


def minkowski_distance_point_to_data(point, data, p=2):
    distance = []
    for data_point in data:
        data_point = np.asarray(data_point)
        distance.append(((sum((i - k) ** p for i, k in zip(data_point, point))) ** 1 / p))
    return distance


def predict(image, average, eigen_face, weight, labels):
    unknown_avg = image - average
    unknown_omega = []
    for value in eigen_face:
        unknown_omega.append(np.dot(np.transpose(value), unknown_avg))

    unknown_omega = np.asarray(unknown_omega)
    number = int(np.floor(
        np.argmin(
            minkowski_distance_point_to_data(unknown_omega, weight))
        * len(labels))
                 / len(weight))

    return labels[number]


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def performance_measure(eigenface_data, train_labels, val_data, val_label):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    confusion_matrix = {k: {p: 0 for p in val_label} for k in train_labels}
    for val, label in zip(val_data, val_label):
        pred = predict(val, eigenface_data['average'], eigenface_data['eigenface'], eigenface_data['weight'],
                       train_labels)
        confusion_matrix[label][pred] += 1
        if label == pred:
            tp += 1
        elif label != pred:
            tn += 1
    print(confusion_matrix)
    print('=' * 10)
    print('acc: {}'.format(tp / len(val_data)))
