import numpy as np
import scipy
import signal
import time
import glob
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from default import *

verti = [1,2,3,4,5]
x_pos = np.arange(5)
name = ['a', 'b', 'c', 'd', 'e']


plt.bar(x_pos, verti, width = 0.35, align = 'center',alpha = 1)
plt.xticks(x_pos, name)
plt.show()
