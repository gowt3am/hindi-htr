import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImage(img, num = 1):           # 0 for Image, 1 for Plot
    if num == 0:
        cv2.imshow('Image', img)
        cv2.waitKey(0)                 #Waits indefinitely for keypress
        cv2.destroyAllWindows()
    elif num == 1:
        plt.imshow(img, cmap='gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  #Hides tick values on X & Y
        plt.show()

def plotHist(hist, num):           # 0 for Col, 1 for Row
    if num == 0:
        hist = np.transpose(hist)
    fig = plt.figure()
    ax = plt.axes()
    x = range(hist.shape[0])
    ax.plot(x, hist);