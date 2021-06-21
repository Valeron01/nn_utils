import numpy as np
import cv2 as cv2
import os
import image_loader


def loadImage(path, w=-1, h=-1, bw=False, flat=False):
    if (not bw) and flat:
        raise Exception('Error: BGR Image can not be flat')
    if bw:
        if not flat:
            if w == -1 or h == -1:
                return cv2.imread(path, 0) / 255
            return np.reshape(cv2.resize(cv2.imread(path, 0), (w, h))/255, (w, h, 1))
        raise NotImplementedError('srry')

    else:
        if w == -1 or h == -1:
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
    folders = [f for f in os.listdir(
        path) if not os.path.isfile(os.path.join(path, f))]
    for i in folders:
        renameImages(path + '\\' + i)


def createDataset(path, bw=False, flat=False, w=-1, h=-1, returnDict=False):
    retx = []
    rety = []
    folders = [f for f in os.listdir(
        path) if not os.path.isfile(os.path.join(path, f))]
    for n, i in enumerate(folders):
        folder = path + '\\' + i
        imgs = [f for f in os.listdir(
            folder) if os.path.isfile(os.path.join(folder, f))]
        y = np.zeros(len(folders))
        y[n] = 1
        for i in imgs:
            image = folder + '\\' + i
            retx.append(loadImage(image, w, h, bw, flat))
            rety.append(y)
    if returnDict:
        return np.array(retx), np.array(rety), folders
    return np.array(retx), np.array(rety)


def shuffle(x, y):
    retx = []
    rety = []

    if len(x) != len(y):
        raise Exception('Array\'s legths not equal!!: X: ' +
                        str(len(x)) + ' Y: ' + str(len(y)))

    l = len(x)
    indexes = []
    while len(indexes) < l:
        n = np.random.randint(l)
        if not n in indexes:
            indexes.append(n)

    for i in indexes:
        retx.append(x[i])
        rety.append(y[i])

    return np.array(retx), np.array(rety)


def getAnswer(y, voc=None):
    i = np.argmax(y)

    if voc == None:
        return i
    return voc[i]


def evaluateModel(ypred, y, logging=False):
    right = 0
    ran = range(len(y))
    a = []
    for i, j, k in zip(ypred, y, ran):
        if getAnswer(i) == getAnswer(j):
            right += 1
        else:
            if logging:
                a.append(k)
    percent = right / len(y)
    if logging:
        return percent, right, len(y) - right, a
    return percent, right, len(y) - right


def predict(model, singleX):
    return model.predict(np.array([singleX]))


def loadAllImages(folder, bw=False, w=-1, h=-1):
    ret = []
    imgs = [f for f in os.listdir(
        folder) if os.path.isfile(os.path.join(folder, f))]
    for i in imgs:
        image = folder + '\\' + i
        ret.append(loadImage(image, w, h, bw, flat))
    return np.array(ret)


def draw_mask(image, mask, fill_color):
    ret, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    mask = np.expand_dims(mask, -1) / 255
    mask_i = 1 - mask

    colored = mask * image * fill_color / 255

    result = colored + mask_i * image
    return np.uint8(result)


def draw_mask_with_contour(image, mask, fill_color, contour_color):

    ret, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(mask, 127, 255) / 255
    canny = np.expand_dims(canny, -1)

    mask = np.expand_dims(mask, -1) / 255 - canny

    mask_i = 1 - mask - canny

    colored = mask * image * fill_color / 255

    contour = canny * image * contour_color / 255

    result = colored + mask_i * image + contour
    return np.uint8(result)
