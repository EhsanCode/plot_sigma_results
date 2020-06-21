#!/usr/bin/env python

import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import SaliencyLib

resize_factor = 4

def main():
    file_path = '061.mat'
    mat = loadmat(file_path)
    gaze_image = mat['fixLocs']
    full_size = np.array(gaze_image.shape)
    gaze_image_resize = np.zeros((full_size//resize_factor).astype(int))
    gaze_full_size_indices = np.array(np.where(gaze_image == 1))
    gaze_indices = (gaze_full_size_indices//resize_factor).astype(int)
    for indice in range(gaze_indices.shape[1]):
        i=gaze_indices[0][indice]
        j=gaze_indices[1][indice]
        gaze_image_resize[i][j] = 1

    for sigma in range(5, 251, 5):
        output_path = 'saliency_sigma_s_' + str(sigma ) + '.png'        
        saliency_map = SaliencyLib.get_saliency_map(gaze_image_resize, sigma)
        SaliencyLib.show_save_image(saliency_map, output_path)


if __name__ == '__main__':

    main()
