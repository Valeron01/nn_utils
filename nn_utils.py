import numpy as np
import cv2 as cv2
import os

def loadImage(path, w=-1, h=-1, bw=False, flat=False):
    
    if bw and flat:
        raise Exception('Error: BGR Image can not be flat')
    if bw:
        if not flat:
            if w==-1 or h == -1:
                return cv2.imread(path) / 255
            return cv2.resize(cv2.imread(path), (w, h))/255
        raise NotImplementedError('srry')

    else:
        if w==-1 or h == -1:
            return cv2.imread(path) / 255
        return cv2.resize(cv2.imread(path), (w, h))/255

def renameImages(path):
    from os import listdir
    from os.path import isfile, join
    imgs = [f for f in listdir(path) if isfile(join(path, f))]
    try:
        for i, im in enumerate(imgs):
            os.rename(path + '\\' + im, path + '\\' + str(i) + '.png')
    except Exception as e:
        print(str(e))

def renameDataset(path):
    folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    for i in folders:
        renameImages(path + '\\' + i)

def createDataset(path, bw=False, flat=False, w=-1, h=-1):
    retx = []
    rety = []
    folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    for n, i in enumerate(folders):
        folder = path + '\\' + i
        imgs = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        y = np.zeros(len(folders))
        y[n] = 1
        for i in imgs:
            image = folder + '\\' + i
            retx.append(loadImage(image, w , h, bw, flat))
            rety.append(y)
    return np.array(retx), np.array(rety)
    