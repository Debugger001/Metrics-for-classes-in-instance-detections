import numpy as np
import scipy
import signal
import time
import glob
import sys
import os
import shutil
from tqdm import tqdm
from xml.dom.minidom import parse
import xml.dom.minidom
import codecs

def read_xml_file(xml_file):
    instances = []
    DOMTree = xml.dom.minidom.parse(xml_file)
    annotation = DOMTree.documentElement
    objects = annotation.getElementsByTagName("object")
    for object in objects:
        instance = []
        name = object.getElementsByTagName("name")[0]
        insclass = name.childNodes[0].data
        # print(insclass)
        bbox = object.getElementsByTagName("bndbox")[0]
        xmax = bbox.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = bbox.getElementsByTagName("ymax")[0].childNodes[0].data
        xmin = bbox.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = bbox.getElementsByTagName("ymin")[0].childNodes[0].data
        # print("bbox: ")
        # print(str(xmax))
        # print(str(xmin))
        # print(str(ymax))
        # print(str(ymin))
        # break
        instance.append(insclass)
        instance.append(int(xmax))
        instance.append(int(ymax))
        instance.append(int(xmin))
        instance.append(int(ymin))
        instances.append(instance)
    return instances
# xml_file = '/Users/pro/Desktop/Lab/Dataset/Pascal-VOC/2012/VOCdevkit/VOC2012/Annotations/2012_002004.xml'
# read_xml_file(xml_file)
