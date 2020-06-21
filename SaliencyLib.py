#!/usr/bin/env python

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt

def _get_gaussian_distribution(sigma):
    half_width = 3 * sigma
    grid_numbers = np.array(range(-half_width, half_width + 1))
    xx, yy = np.meshgrid(grid_numbers, grid_numbers)
    gaussian_distribution = np.exp(-1/(2*sigma ^ 2)
                                   * (np.power(xx, 2) + np.power(yy, 2)))
    return gaussian_distribution


def get_saliency_map(gaze_image, sigma):
    gaussian_distribution = _get_gaussian_distribution(sigma)
    saliency_map = signal.convolve2d(
        gaze_image, gaussian_distribution, boundary='symm', mode='same')
    saliency_map /= saliency_map.max()
    return saliency_map


def show_save_image(img, output_path):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    fig.savefig(output_path)
    # plt.show()
