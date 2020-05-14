import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def pageSegment(doc, Hfrac = 0.1, Wfrac = 0.01, 
               DEkernel_size = (2,2), blur_rad = 3, 
               max_ang = 2.1, delta = 0.05,
               noise = False, blur = True, binary = 'THRESH_OTSU',
               skew = True):
    doc = borderDel(doc, Hfrac, Wfrac)
    if noise == True:
        doc = noiseRem(doc, DEkernel_size, ite)
    if blur == True:
        doc = gaussBlur(doc, blur_rad)
    doc = binarize(doc, binary)
	doc = 255 * (1 - doc)
    if skew == True:
        doc, _ = skCorr(doc, max_ang, delta)
	page = pageCrop(doc)
    return page
	
def borderDel(img, Hfrac = 0.1, Wfrac = 0.01):
    Hcut = int(Hfrac * img.shape[0]/2)
    Wcut = int(Wfrac * img.shape[1]/2)
    return img[Hcut:img.shape[0] - Hcut, Wcut:img.shape[1] - Wcut]

def noiseRem(img, DEkernel_size = (2,2), ite = 1):
    DEkernel = np.ones(DEkernel_size, np.uint8)
    img = cv2.dilate(img, DEkernel, iterations = 1)
    img = cv2.erode(img, DEkernel, iterations = ite)
    return img
    
def gaussBlur(img, blur_rad = 3):
    shape = (blur_rad, blur_rad)
    img = cv2.GaussianBlur(img, shape, 0)
    return img
    
def binarize(img, method = 'THRESH_OTSU'):
    if method == 'THRESH_BINARY':
        img = cv2.threshold(img, 210, 1, cv2.THRESH_BINARY)[1]
    elif method == 'THRESH_BINARY_INV':
        img = cv2.threshold(img, 210, 1, cv2.THRESH_BINARY_INV)[1]
    elif method == 'THRESH_OTSU':
        img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    elif method == 'ADAPTIVE_GAUSSIAN':
        img = 1 - cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'ADAPTIVE_MEAN':
        img = 1 - cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return img

def skCorr(img, max_ang = 2.1, delta = 0.05):
    angles = np.arange(-max_ang, max_ang+delta, delta)
    scores = []
    hists = []
	temp = img[int(0.06*img.shape[0]):][:]
    for angle in angles:
        hist, score = findScore(temp, angle)
        scores.append(score)
        hists.append(hist)
        
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Skew Angle: ' + str(best_angle))
    data = inter.rotate(img, best_angle, reshape = False, order = 0, cval = 255)
    return data, best_angle

def findScore(arr, angle):
    data = inter.rotate(arr, angle, reshape = False, order = 0, cval = 255)
	data = (255 - data)/255
    hist = np.sum(data, axis = 1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def pageCrop(img):
    hist = (np.sum(img, axis = 0)/img.shape[0], np.sum(img, axis = 1)/img.shape[1])
    bottomPeak = returnBottomPeak(img, hist[1])
	topPeak = returnTopPeak(img, hist[1])
    leftPeak, rightPeak = returnLeftRightPeak(img, hist[0])
    newImg = img[topPeak:bottomPeak, int(leftPeak):int(rightPeak)]
    return img

def returnTopPeak(img, hist):
    peaks = []
    i = 1
    while i<hist.shape[0]/2:
        if hist[i-1] <= hist[i] and hist[i]>=hist[i+1]:
            peaks.append((i,hist[i]))
            i = i + 3
        else:
            i = i + 1
    peak1 = peaks[0]
    peak2 = None
    for i in peaks:
        if i[1]>peak1[1]:
            peak2 = peak1
            peak1 = i
        elif peak2 == None or peak2[1]<i[1]:
            peak2 = i
    coord = peak1[0] if peak1[0]>peak2[0] else peak2[0]
    
    c01 = int(coord - 20)
    c02 = int(coord + 20)
    temp = img[c01:c02,:]
    histo = np.sum(temp,axis=0)/(c02-c01)
    count = 0
    for j in range(histo.shape[0]):
        if histo[j] < 255 and histo[j] > 220:
            count = count+1

    if count/histo.shape[0] > 0.75:
        value = int(coord+0.003*img.shape[0])
    else:
        return int(coord-0.02*img.shape[0])
    
	hist = 255 - hist
    val = value
    while val < img.shape[0]:
        if hist[val] > 250:
            val += 1
        else:
            print('Top Crop:' + str(val))
            return max(value,int(val - 0.01*img.shape[0]))
    return value
    

def returnBottomPeak(img,hist):
    i = img.shape[0]
    val = img.shape[0]
    while i>img.shape[0]*0.55:
       # flag = True
        count = 0
        tem = img[i-50:i]
        histo = np.sum(tem,axis = 0)/50
        for j in range(tem.shape[1]):
            if histo[j] < 250:
                count = count+1
        if count/img.shape[1] > 0.75:
            val = min(int(i-25 - 0.025*img.shape[0]),hist.shape[0]-1)
        i = i - 50
        
    while val > 0:
        if hist[val] > 250:
            val -= 1
        else:
            print('Bottom Crop:' + str(val))
            return int(val + 0.025*img.shape[0])

def returnLeftRightPeak(img, hist):
    (left, right) = (0, img.shape[1]-1)
    i = img.shape[1]-1
    while i>img.shape[1]*0.7:
        count = 0
        tem = img[:,i-20:i]
        histo = np.sum(tem,axis = 1)/20
        for j in range(tem.shape[0]):
            if histo[j] < 250:
                count = count+1
        if count/img.shape[0] > 0.45:
            right = min(int(i-25 - 0.025*img.shape[1]),hist.shape[0]-1)
        i = i - 20
      
    i = 0
    while i<img.shape[0]*0.3:
        count = 0
        tem = img[:,i:i+20]
        histo = np.sum(tem,axis = 1)/20
        for j in range(tem.shape[0]):
            if histo[j] < 250:
                count = count+1
        if count/img.shape[0] > 0.45:
            left = int(i+25)
        i = i + 20
        
    while left < img.shape[1]:
        if hist[left] > 253:
            left += 1
        else:
            print('Left Crop:' + str(left))
            left = max(0,left - 0.0015*img.shape[1])
            break
        
    while right > 0:
        if hist[right] > 253:
            right -= 1
        else:
            print('Right Crop:' + str(right))
            right = right + 0.02*img.shape[1]
            break
        
    return left, right
    