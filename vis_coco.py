"""

this script will using pycoco API
draw our converted annotation to check
if result is right or not

"""
from pycocotools.coco import COCO
import os
import sys
import cv2
from pycocotools import mask as maskUtils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import skimage.io as io


data_dir = './cn_images_20190827'
ann_f = 'annotation/instances_train2014.json'
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
    ann_f = sys.argv[2]

coco = COCO(ann_f)

cats = coco.loadCats(coco.getCatIds())
print('cats: {}'.format(cats))

img_ids = coco.getImgIds()
print('img_ids: {}'.format(img_ids))


for i in range(9):
    img = coco.loadImgs(img_ids[i])
    print('checking img: {}, id: {}'.format(img, img_ids[i]))
    img_f = os.path.join(data_dir, img[0]['file_name'])

    # draw instances
    anno_ids = coco.getAnnIds(imgIds=img[0]['id'])
    annos = coco.loadAnns(anno_ids)

    I = io.imread(img_f)
    plt.imshow(I)
    plt.axis('off')

    coco.showAnns(annos)
    plt.show()


