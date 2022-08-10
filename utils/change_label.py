""" change colors of labels
-----------------------------------------------------------------------
Typical usage example:

    Change white label to red label when merging two datasets

-----------------------------------------------------------------------
Usage:
    params.json
    More see README.md
----------------------------------------------------------------------
"""

import os
from glob import glob
import natsort
import numpy as np
from PIL import Image
import json

def old2new(image, old_label, new_label):

    # Convert 3d to 2d
    mask = np.zeros((image.shape[0], image.shape[1]))
    for k, v in old_label.items():
        mask[np.all(image == v, axis=2)] = int(k)

    # Convert 2d to 3d
    newColor = np.zeros(mask.shape+(3,), dtype=np.uint8)
    for i in np.unique(mask):
        newColor[mask == i] = new_label[i]

    return newColor


if __name__ == '__main__':

    PATH_PARAMETERS = './params.json'
    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    change_label_settings = params['change_label']

    old_label_dict = change_label_settings["oldLabel"]
    old_label = {}
    for i, value_correspond in enumerate(old_label_dict.values()):
        value_RGB = list(value_correspond.values())
        value_RGB = value_RGB[0]
        temp = {i: value_RGB}
        old_label.update(temp)

    new_label_dict = change_label_settings["newLabel"]
    new_label = {}
    for i, value_correspond in enumerate(new_label_dict.values()):
        value_RGB = list(value_correspond.values())
        value_RGB = value_RGB[0]
        temp = {i: value_RGB}
        new_label.update(temp)


    # Initialize the image path
    path = change_label_settings["mask_folder"]
    path_masks = natsort.natsorted(glob(os.path.join(path, '*.png')))

    for idx in range(len(path_masks)):

        # read picture
        mask_read = Image.open(path_masks[idx])

        # Transform old color to new color
        mask = old2new(np.array(mask), old_label, new_label)

        # Transform matrix to image format
        mask = Image.fromarray(mask)

        # save
        save_folder = change_label_settings["save_folder"]
        mask.save(save_folder+str(idx)+'.png', 'png')
