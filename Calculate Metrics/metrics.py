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
from get_bbox import read_xml_file

def calculate_precision_recall(img_file):
    '''
        Args: the file name of the images, without the file extension.
    '''
    TP = [0] * len(COLOR_DIC)
    FP = [0] * len(COLOR_DIC)
    FN = [0] * len(COLOR_DIC)
    P = [0] * len(COLOR_DIC)
    precision = [0] * len(COLOR_DIC)
    recall = [0] * len(COLOR_DIC)
    # print(TP)
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
        P[ORDER_DIC[instance_class]] = 1
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
        # print(instance_class)
        # print(iou)
    for i in range(len(COLOR_DIC)):
        if TP[i] + FP[i] > 0:
            precision[i] = TP[i] / (TP[i] + FP[i])
            # recall[i] = TP[i] / (TP[i] + FN[i])

    # cv2.imwrite("test.jpg", sem_img_np)

    return precision, recall, P

def calculate_miou():
    IMG_PATH = PASCAL_VOC2012_DIR + SEM_MASK_DIR
    list = os.listdir(IMG_PATH)
    # print(list)
    precisions = []
    recalls = []
    Ps = []
    # count = 0
    for img in tqdm(list):
        # count += 1
        # if (count >= 50):
        #     break
        img_file = img.split('.')[0]
        if img_file == '' or img_file == 'pre_encoded':
            continue
        precision = []
        recall = []
        precision, recall, P = calculate_precision_recall(img_file)
        # print(precision)
        precisions.append(precision)
        recalls.append(recall)
        Ps.append(P)
    mIoUs = []
    for c in range(len(COLOR_DIC)):
        sum_prec = 0
        img_num = 0
        for i in range(len(precisions)):
            img_num += Ps[i][c]
            sum_prec += precisions[i][c]
        mIoU = 0
        if img_num > 0:
            mIoU = sum_prec / img_num
        mIoUs.append(mIoU)
    return precisions, recalls, mIoUs

def visualization(mIoUs):
    x_pos = np.arange(len(mIoUs))
    x_name = ["aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"]
    # print(histboxnum)
    plt.bar(x_pos, mIoUs, align = 'center', alpha = 1, color = 'lightskyblue')
    plt.xlabel('classes', fontsize = 12)
    plt.ylabel('precision', fontsize = 12)
    plt.title('precisions', fontsize = 12)
    plt.xticks(x_pos, x_name)
    for x,y in zip(x_pos,mIoUs):
        plt.text(x+0.3, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
    set(gca,'XTickLabelRotation',90)
    fig = plt.gcf()
    fig.set_size_inches(7.2, 4.2)
    fig.savefig('Precisions.png', dpi=100)
    plt.show()


# calculate_precision_recall('2007_000129')
precisions, recalls, mIoUs = calculate_miou()
visualization(mIoUs)
print(mIoUs)
# print(len(precisions))
# print(precisions[0])
