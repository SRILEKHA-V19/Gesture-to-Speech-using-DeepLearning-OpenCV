import cv2
import numpy as np


def pre_process_images(path_name, file, count):
    image = cv2.imread(path_name)

    # Skin coloured portions of image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    # Convert grayscale image to binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', thresh)

    cv2.imwrite("/Users/svinjamara/Documents/Preprocess_5/{}_{}.png".format(file, count), thresh)

