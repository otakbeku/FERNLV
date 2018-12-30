import numpy as np
import cv2
import os
import re
import shutil
import hashlib
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
import pickle

# TODO this project still used tensorflow module which  not always available for everyone. Used with cautions.

EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
MAX_NUM_IMAGES_PER_CLASS = 100
# For the use of face cascade please refers to:
#  https://stackoverflow.com/questions/4440283/how-to-choose-the-cascade-file-for-face-detection
# Which is tell the default is the best classifier
# yet haven't compared with LBP and cuda version
FACE_CASCADES = {
    'default': cv2.CascadeClassifier(
        '../Assets/haarcascade_frontalface_default.xml'),
    'alt': cv2.CascadeClassifier(
        '../Assets/haarcascade_frontalface_alt.xml'),
    'alt2': cv2.CascadeClassifier(
        '../Assets/haarcascade_frontalface_alt2.xml'),
    'altree': cv2.CascadeClassifier(
        '../Assets/haarcascade_frontalface_alt_tree.xml'),
    'profile': cv2.CascadeClassifier(
        '../Assets/haarcascade_profileface.xml'),
    'LBP': cv2.CascadeClassifier(
        '../Assets/lbpcascade_frontalface_improved.xml'),
    'cudaDef': cv2.CascadeClassifier(
        '../Assets/haarcascade_frontalface_default_cuda.xml'
    )
}

CASCADE_DEFAULT = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def get_face_cascades(name='default'):
    """
    Demystify the Face Cascade Classifier from OpenCV

    :param
        name: name of haarcascade that available from OpenCV which is:
            'default': default frontal face,
            'alt': alternative of frontal face,
            'alt2': another alternative for frontal face,
            'altree': another alternative version with tree ,
            'profile': Only detect the profile face,
            'LBP': XMl made with Local Binary Pattern,
             'cudaDef': A Cuda version of default

    :return:
        Chosen face cascade classifier
    """
    return FACE_CASCADES[name]


def vec_normalization(vectors):
    """
    Normalize the vectors with min-max normalization

    :param vectors: numpy array
    :return: normalized vector
    """
    vectors = np.asarray(vectors)
    max_pixel = vectors.max()
    min_pixel = vectors.min()
    for index in range(len(vectors)):
        new_pixel = []
        for indexPixel in range(len(vectors[index])):
            value = 255 * (vectors[index, indexPixel] - min_pixel) / (max_pixel - min_pixel)

            value = np.uint8(value.real)
            new_pixel.append(value)
        vectors[index] = new_pixel
    return vectors


def get_cropped_face(image,
                     y_plus=0, x_plus=0,
                     face_cascade: cv2.CascadeClassifier =
                     FACE_CASCADES['default']):
    """
    A method to crop a (first) detected face from an image using face cascade classifier from OpenCV

    :param image: an image, numpy ndarray
    :param y_plus: margin of y
    :param x_plus: margin of x
    :param face_cascade: a face cascade classifier. The default value is default haarcascade XML file
    :return: cropped face, (x, y, w, h)
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    d = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    try:
        [[x, y, w, h]] = d[0]
        crop = image_gray[y - y_plus:y + h + y_plus, x - x_plus:x + w + x_plus]
        return crop, (x, y, w, h)

    except Exception as e:
        print("Err: ", str(e))
        return None, (0, 0, 0, 0)


# TODO fix this so it can return the proper value
def get_cropped_faces(image,
                      y_plus=0, x_plus=0,
                      face_cascade: cv2.CascadeClassifier =
                      FACE_CASCADES['default']):
    """
    A better version of getting the cropped face. It will return the several faces
    :param image: an image, numpy ndarray
    :param y_plus: margin of y
    :param x_plus: margin of x
    :param face_cascade: a face cascade classifier. The default value is default haarcascade XML file
    :return: cropped faces, (x, y, w, h)
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    d = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    try:
        for x, y, w, h in d:
            crop = image_gray[y - y_plus:y + h + y_plus, x - x_plus:x + w + x_plus]
            return crop, (x, y, w, h)

    except Exception as e:
        print("Err: ", str(e))
        return 0, (0, 0, 0, 0)


def image_to_vec(image: np.ndarray, size):
    """
    Make image to vector by resize it first into (size, size)

    :param image: image
    :param size: desired size of image
    :return: a vector with size*size length
    """
    temp = cv2.resize(image, (size, size))
    vec = temp.flatten()
    return vec


def split_data_train_test_val(image_dir: str, base_dir: str = 'base_dir',
                              testing_percentage=0,
                              validation_percentage=10,
                              size=32, save=True,
                              cropped=False):
    """
    This method is based from: https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/6be494e0300555fd48c095abd6b2764ba4324592/scripts/retrain.py#L125
    to split the data to 3-different purpose with a little addition

    :param image_dir: a root folder consist of category-named directories. The root directory should looks like this:
         -> root_dir:
             --> class_1
             --> class_2
             --> ...
    :param base_dir: the directory to placed the train, validation and testing.
    :param testing_percentage: From the available data, it will used n-percent for testing. USE INTEGER
    :param validation_percentage:From the available data, it will used n-percent for validation. USE INTEGER
    :param size: The size to resize the image. Suggested 32, so the image will resize into (32, 32)
    :param save: Save option. If True, it will produce pickle files for train, validation and testing (if available)
    :param cropped: if the dataset already in desired condition, put this into False. Otherwise it will cropped faces
    from the image using default face cascade classifier. Use it with cautions
    :return: train_data, train_labels, val_data, val_labels, test_data, test_labels
    """
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
            temp = cv2.imread(file_name, 0)
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
    """
    A method to produce a weight from eigen vector with its subtracted average

    :param eigen_face: the eigenface
    :param subtracted_average: subtracted average from the base of eigenface
    :return: weight
    """
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
    """
    Method to produce eigenface vector

    :param data: data that consists of image vector in each row
    :param normalization: the option for normalization
    :param max_component: number of component to be used for eigenface
    :param save: the option for saving the eigenface
    :return: eigen_face, weight, average
    """
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
    """
    This method exclusively made for this project. This would compute the distance between the given point to the data

    :param point: the given point or vector
    :param data: the known data
    :param p: 1 for manhattan distance, 2 for euclidean distance
    :return: distance
    """
    distance = []
    for data_point in data:
        data_point = np.asarray(data_point)
        distance.append(((sum((i - k) ** p for i, k in zip(data_point, point))) ** (1/p)))
    return distance


def predict(image, average, eigen_face, weight, labels, dist_thredhold=9291122):
    """
    A Method that would return a predict of a given image
    :param image: the image must be 80% covered with face
    :param average: average of the data
    :param eigen_face: eigeface that generated from the data
    :param weight: weight fron eigenface and subtracted averaga
    :param labels: labels of the data
    :param dist_thredhold: a distance threshold that would identified
    :return: known label or unknown
    """
    unknown_avg = image - average
    unknown_omega = []
    for value in eigen_face:
        unknown_omega.append(np.dot(np.transpose(value), unknown_avg))
    unknown_omega = np.asarray(unknown_omega)
    distance_num = np.argmin(minkowski_distance_point_to_data(unknown_omega, weight))
    if distance_num <= dist_thredhold:
        number = int(np.floor(distance_num * len(labels)) / len(weight))
        return labels[number]
    else:
        return 'unknown'


# TODO this should be in the utils
def load_pickle(path):
    """
    Just load the pickle from the given path
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# Todo this just a dummy version of performance measure. Should adding some k-fold validation
def performance_measure(eigen_face, train_labels, val_data, val_label):
    """
    A simple performance measure that only compute the accuracy.
     Precision, recall and F1 Score is not available for now.
     The data (validation and training) should be in the format (flatten-image matrix).

    :param eigen_face: the eigenface of the data
    :param train_labels: the label of the data
    :param val_data: the validation data
    :param val_label: the label of the validation data
    :return: Nothing. Just print the result
    """
    tp = 0
    tn = 0
    confusion_matrix = {k: {p: 0 for p in val_label} for k in train_labels}
    for val, label in zip(val_data, val_label):
        pred = predict(val, eigen_face['average'], eigen_face['eigenface'], eigen_face['weight'],
                       train_labels)
        confusion_matrix[label][pred] += 1
        if label == pred:
            tp += 1
        elif label != pred:
            tn += 1
    print(confusion_matrix)
    print('=' * 10)
    print('acc: {}'.format(tp / len(val_data)))
