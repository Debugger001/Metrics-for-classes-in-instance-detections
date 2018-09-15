import numpy as np
import scipy
import signal
import time
import glob
import sys
import os
import default

def calculate_iou(img_file):
    '''
        Args: the file name of the images, without the file extension.
    '''
    sem_mask_file = PASCAL_VOC2012_DIR + SEM_MASK_DIR + '/' + img_file + '.png'
