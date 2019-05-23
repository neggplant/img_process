# return images and label(index of forder)

import os
from skimage import io,transform
import numpy as np
import glob
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    print(cate)
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)

            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs), np.asarray(labels,np.int32)
