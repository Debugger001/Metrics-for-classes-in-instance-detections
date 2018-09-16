import numpy as np
import scipy
import signal
import time
import glob
import sys
import os
from PIL import Image
import cv2
from default import *
from get_bbox import read_xml_file

def calculate_precision_recall(img_file):
    '''
        Args: the file name of the images, without the file extension.
    '''
    TP = [0] * len(COLOR_DIC)
    FP = [0] * len(COLOR_DIC)
    FN = [0] * len(COLOR_DIC)
    precision = [0] * len(COLOR_DIC)
    recall = [0] * len(COLOR_DIC)
    print(TP)
    sem_mask_file = PASCAL_VOC2012_DIR + SEM_MASK_DIR + '/' + img_file + '.png'
    # sem_img = Image.open(sem_mask_file)
    sem_img_np = cv2.imread(sem_mask_file)
    # print(sem_img.getpixel((0,10)))
    # sem_img_np = np.array(sem_img.getdata(), dtype = np.uint8).reshape(sem_img.size[1], sem_img.size[0], 3)
    annotation_file = PASCAL_VOC2012_DIR + ANNOTATION_DIR + '/' + img_file + '.xml'
    instances = read_xml_file(annotation_file)
    for instance in instances:
        obj_pixel = 0
        instance_class, xmax, ymax, xmin, ymin = instance
        instance_color = COLOR_DIC[instance_class]
        r,g,b = instance_color
        for j in range(xmin, xmax):
            for i in range(ymin, ymax):
                if sem_img_np[i][j][0] == b and sem_img_np[i][j][1] == g and sem_img_np[i][j][2] == r:
                    obj_pixel += 1
        iou = obj_pixel / ((xmax-xmin)*(ymax-ymin))
        if iou > THRESHOLD:
            # this is a true positive (TP)
            TP[ORDER_DIC[instance_class]] += 1
        if iou < THRESHOLD:
            FP[ORDER_DIC[instance_class]] += 1
        # for jj in range(xmin, xmax):
        #     sem_img_np[ymin][jj] = (255, 255, 255)
        #     sem_img_np[ymax - 1][jj] = (255, 255, 255)
        # for ii in range(ymin, ymax):2
        #     sem_img_np[ii][xmin] = (255, 255, 255)
        #     sem_img_np[ii][xmax - 1] = (255, 255, 255)
        print(instance_class)
        print(iou)
    for i in range(len(COLOR_DIC)):
        if TP[i] + FP[i] > 0:
            precision[i] = TP[i] / (TP[i] + FP[i])
            recall[i] = TP[i] / (TP[i] + FN[i])

    # cv2.imwrite("test.jpg", sem_img_np)

    return precision, recall

def calculate_ap():
    IMG_PATH = PASCAL_VOC2012_DIR + SEM_MASK_DIR
    list = os.listdir(IMG_PATH)
    precisions = []
    recalls = []
    for img in list:
        img_file = img.split('.')[0]
        precision = []
        recall = []
        precision, recall = calculate_precision_recall(img_file)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls


# calculate_precision_recall('2007_000129')
precisions, recalls = calculate_ap()
print(len(precisions))
print(precisions[0])
