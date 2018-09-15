import numpy as np
import scipy
import signal
import time
import glob
import sys
import os

def calculate_iou(img_file):
    '''
        Args: the file name of the images, without the file extension.
    '''
    sem_mask_file = img_file + '.png'
