import numpy as np
import matplotlib as plt
import os
import SimpleITK as sitk
import seaborn as sns
import pandas as pd



def dice_coefficient(seg1, seg2):
    intersection = (np.logical_and(seg1, seg2)).sum()
    union = seg1.sum() + seg2.sum()
    return 2 * intersection / union



