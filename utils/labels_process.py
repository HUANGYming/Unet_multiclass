""" Convert 3-D labels to 2-D labels or vice versa

Usage: 
    See README.md

"""
import numpy as np

# Corresponding relationship of the label numbers and the RGB value

class labels_process():
    
    def __init__(self, train_settings):

        LABEL_TO_COLOR_DICT = train_settings["label_to_value"]

        self.LABEL_TO_COLOR = {}
        for i, value_correspond in enumerate(LABEL_TO_COLOR_DICT.values()):
            value_RGB = list(value_correspond.values())
            value_RGB = value_RGB[0]
            temp = {i: value_RGB}
            self.LABEL_TO_COLOR.update(temp)

    # Convert 2-D labels to 3-D labels
    def mask2rgb(self, mask):
        
        rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
        
        for i in np.unique(mask):
            rgb[mask==i] = self.LABEL_TO_COLOR[i]

        return rgb

    # Convert 3-D labels to 2-D labels
    def rgb2mask(self, rgb):
        
        mask = np.zeros((rgb.shape[0], rgb.shape[1]))
        
        for k,v in self.LABEL_TO_COLOR.items():
            mask[np.all(rgb==v, axis=2)] = k

        return mask

