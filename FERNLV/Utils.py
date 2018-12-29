from PyQt5.Qt import QImage


def convert_to_QImage(image):
    """
    Convert from numpy ndarray or OpenCV image to QImage for PyQt5
    :param image: an image. This should be in 3-color channel
    :return: the RGB Image. R-G-B respectively
    """
    image_format_qt = QImage.Format_Indexed8
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image_format_qt = QImage.Format_RGBA8888
        else:
            image_format_qt = QImage.Format_RGB888

    rgb_image = QImage(image, image.shape[1], image.shape[0], image.strides[0], image_format_qt)
    rgb_image = rgb_image.rgbSwapped()
    return rgb_image
