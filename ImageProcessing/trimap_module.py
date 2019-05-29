#!/usr/bin/env python
import cv2, os, sys
import numpy as np


def extractImage(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image


def checkImage(image):
    """
    Args:
        image: input image to be checked

    Returns:
        binary image

    Raises:
        RGB image, grayscale image, all-black, and all-white image

    """
    if len(image.shape) > 2:
        print("ERROR: non-binary image (RGB)")
        sys.exit()

    smallest = image.min(axis=0).min(axis=0)
    # lowest pixel value; should be 0 (black)
    largest = image.max(axis=0).max(axis=0)
    # highest pixel value; should be 1 (white)

    if (smallest == 0 and largest == 0):
        print("ERROR: non-binary image (all black)")
        sys.exit()
    elif (smallest == 255 and largest == 255):
        print("ERROR: non-binary image (all white)")
        sys.exit()
    elif (smallest > 0 or largest < 255):
        print("ERROR: non-binary image (grayscale)")
        sys.exit()
    else:
        return True


def trimap(image, size, erosion=False):
    """
    This function creates a trimap based on simple dilation algorithm
    Inputs [3]: a binary image (black & white only), dilation pixels
                the last argument is optional; i.e., how many iterations will the image get eroded
    Output    : a trimap
    """
    checkImage(image)

    row = image.shape[0]
    col = image.shape[1]

    pixels = 2 * size + 1
    ## Double and plus 1 to have an odd-sized kernel
    kernel = np.ones((pixels, pixels),
                     np.uint8)  ## How many pixel of extension do I get

    if erosion is not False:
        erosion = int(erosion)
        erosion_kernel = np.ones(
            (3, 3), np.uint8)  ## Design an odd-sized erosion kernel
        image = cv2.erode(
            image, erosion_kernel,
            iterations=erosion)  ## How many erosion do you expect
        image = np.where(
            image > 0, 255,
            image)  ## Any gray-clored pixel becomes white (smoothing)
        # Error-handler to prevent entire foreground annihilation
        if cv2.countNonZero(image) == 0:
            print("ERROR: foreground has been entirely eroded")
            sys.exit()

    dilation = cv2.dilate(image, kernel, iterations=1)

    dilation = np.where(dilation == 255, 127, dilation)  ## WHITE to GRAY
    remake = np.where(dilation != 127, 0, dilation)  ## Smoothing
    remake = np.where(image > 127, 200,
                      dilation)  ## mark the tumor inside GRAY

    remake = np.where(remake < 127, 0, remake)  ## Embelishment
    remake = np.where(remake > 200, 0, remake)  ## Embelishment
    remake = np.where(remake == 200, 255, remake)  ## GRAY to WHITE

    #############################################
    # Ensures only three pixel values available #
    # TODO: Optimization with Cython            #
    #############################################
    for i in range(0, row):
        for j in range(0, col):
            if (remake[i, j] != 0 and remake[i, j] != 255):
                remake[i, j] = 127

    return remake
