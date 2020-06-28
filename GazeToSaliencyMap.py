

import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import SaliencyLib

resize_factor = 1

def main():
    file_path = 'data/061.mat'
    image_file_path = 'images/061.jpg'
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

    sigmas = [500]
    for sigma in sigmas:
        print(sigma)
        output_path = 'color_saliency_sigma_s_' + str(sigma ) + '.png'        
        saliency_map = SaliencyLib.get_saliency_map(gaze_image_resize, sigma)
        image = plt.imread(image_file_path) / 255.0
        # image = np.resize(image, heat_map_image.shape)
        # image = np.zeros(heat_map_image.shape)
        image_heatmap = SaliencyLib.apply_heatmap(image, saliency_map)
    
        fig, ax = plt.subplots()
        ax.imshow(image_heatmap, vmin=0, vmax=1)
        ax.axis('off')
        plt.show()
        #SaliencyLib.show_save_image(saliency_map, output_path)


if __name__ == '__main__':

    main()
