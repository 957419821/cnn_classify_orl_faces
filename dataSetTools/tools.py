from PIL import Image
import numpy as np
import os

def getLabels(filedir):
    """ 
    the returned value likes this: 's12',
    the value can be used as a part of path.
    used by function getData(), getFeatures(), getSize()
    """
    labels = os.listdir(filedir)
    for i in range(len(labels)):
        labels[i] = int(labels[i][1:])
    return labels

def getSize(filedir):
    lbs = getLabels(filedir)
    im = Image.open(filedir + 's' + str(lbs[0]) + '/' + os.listdir(filedir + 's' + str(lbs[0]))[0])
    width, height = im.size
    return width, height

def translation(image, vector):
    vertical, horizon = [vector[0, 0], vector[0, 1]]
    vertical = int(vertical); horizon = int(horizon)
    rowLength = len(image); columnLength = len(image[0])
    columnPadding = np.rint(np.matlib.randn((rowLength, np.abs(horizon))) * 4 + 50)
    rowPadding = np.rint(np.matlib.randn((np.abs(vertical), columnLength)) * 4 + 50)
    if horizon > 0:
        image = np.column_stack((columnPadding, image[:, 0:-horizon]))
    elif horizon < 0:
        image = np.column_stack((image[:, -horizon:], columnPadding))
    if vertical > 0:
        image = np.row_stack((rowPadding, image[0:-vertical, :]))
    elif vertical < 0:
        image = np.row_stack((image[-vertical:, :], rowPadding))
    return image

def addNoises(features):
    print('-------------------------------------------------')
    print('the shape of features in function addNoises(): ', np.shape(features))
    print('-------------------------------------------------')
    imagesTiledTimes = 10; labelsHoldOn = imageHeightHoldOn = imageWidthHoldOn = 1
    features = np.tile(features, (labelsHoldOn, imagesTiledTimes, imageHeightHoldOn, imageWidthHoldOn))
    print('-------------------------------------------------')
    print('the shape of tiledFeatures in function addNoises(): ', np.shape(features))
    print('-------------------------------------------------')
    print('-------------------------------------------------')
    print('the type of tiledFeatures in function addNoises(): ', type(features[0]))
    print('-------------------------------------------------')
    #noises = np.rint(np.matlib.randn(np.shape(features)) * sigma)
    imageShape = (112, 92); sigma = 3
    noises = []
    for eachLabel in range(len(features)):
        imageNoises = []
        for eachImage in range(len(features[eachLabel])):
            imageNoise = np.rint(np.matlib.randn(imageShape) * sigma)
            imageNoises.append(imageNoise)
        noises.append(imageNoises)
    noises = np.array(noises)
    features += noises
    features = np.reshape(features, (40, -1, 10304))
    return features

def augment(features, translationTimes=9):
    def getRandomVector():
        return np.rint(np.matlib.rand(2) * 30 - 15)
    print('-------------------------------------------------')
    print('the shape of features in function augment(): ', np.shape(features))
    print('-------------------------------------------------')
    for i in range(len(features)):
        images = []
        for j in range(len(features[i])):
            for k in range(translationTimes):
                images.append(translation(features[i][j], getRandomVector()))
        features[i] += images
    print('-------------------------------------------------')
    print('the shape of translatedFeatures in function augment(): ', np.shape(features))
    print('-------------------------------------------------')
    features = addNoises(features)
    return features


def getFeatures(filedir):
    """
    used by function getData()
    """
    lbs = getLabels(filedir)
    width, height = getSize(filedir)
    features = [os.listdir(filedir + 's' + str(lbs[i])) for i in range(len(lbs))]
    for i in range(len(lbs)):
        for j in range(len(features[i])):
            im = Image.open(filedir + 's' + str(lbs[i]) + '/' + features[i][j]) # type(im): <class 'PIL.PpmImagePlugin.PpmImageFIle'>
            im = im.convert('L')  # type(im): <class 'PIL.Image.Image'>
            data = im.getdata()   # type(data): <class 'ImagingCore'>
            img = np.reshape(list(data), (height, width))
            features[i][j] = img
    return features

def getData(filedir):
    """
    datas, which is returned, contains two parts: features and labels.
    the labels in datas will be formatted in the function.
    """
    lbls = getLabels(filedir)
    for i in range(len(lbls)):
        zeros = [0 for k in range(len(lbls))]
        subscript = lbls[i] - 1
        zeros[subscript] = 1
        lbls[i] = zeros
    print(lbls[0])
    features = getFeatures(filedir)
    print('-------------------------------------------------')
    print('the shape of features in function getData(): ', np.shape(features))
    print('-------------------------------------------------')
    features = list(augment(features))
    datas = []
    for label in range(len(features)):
        features[label] = list(features[label])
        for image in range(len(features[label])):
            features[label][image] = list(features[label][image])
            features[label][image].append(lbls[label])
            datas.append(features[label][image])
    return datas

 
