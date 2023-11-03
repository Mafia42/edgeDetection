# ==============================================================================================================
# This project is to use edge detection on images to find the outline of the contents of the images.
#
#
# Author: Cilliers Schultz
# Start date: 01/11/2023
# ==============================================================================================================

# imports

import numpy as np
import cv2

# A gaussian blur smooths an image.
# A gaussian kernel is used to do this (n, n)
# Larger kernel means bigger blur, and a smaller kernel is a smaller blur


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_filter(size, sigma=1, verbose=False):
    # size is the kernel size. Note, the kernel is basically a matrix and size is n for a nxn matrix
    # print('Starting the gaussian filter')
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1d[i] = dnorm(kernel_1d[i], 0, sigma)
    kernel_2d = np.outer(kernel_1d.T, kernel_1d.T)

    kernel_2d *= 1.0 / kernel_2d.max()

    if verbose:
        pixel_size = 50
        g_filter_image = np.zeros([size * pixel_size, size * pixel_size])

        for i in range(size):
            for j in range(size):
                g_filter_image[i * pixel_size:(i + 1) * pixel_size, j * pixel_size:(j + 1) * pixel_size] = kernel_2d[i, j]

        cv2.imshow("Gaussian filter", g_filter_image)
        cv2.waitKey(0)

    return kernel_2d


def convert_image_array(image):
    array = np.asarray(image)
    return array


if __name__ == '__main__':
    img = cv2.imread('Resources/Bat_for_edgeDetection.jpg')

    print("Creating the Gaussian filter")
    g_filter_size = 15
    std_dev = np.sqrt(g_filter_size)
    g_filter = gaussian_filter(g_filter_size, std_dev, True)
    print("The Gaussian filter has been created")
    # print(g_filter)



    # Creating GUI window to display an image on screen
    # first Parameter is windows title (should be in string format)
    # Second Parameter is image array
    # cv2.imshow("Original image", img)

    # To hold the window on screen, we use cv2.waitKey method
    # Once it detected the close input, it will release the control
    # To the next line
    # First Parameter is for holding screen for specified milliseconds
    # It should be positive integer. If 0 pass a parameter, then it will
    # hold the screen until user close it.
    cv2.waitKey(0)

    # It is for removing/deleting created GUI window from screen
    # and memory
    cv2.destroyAllWindows()
