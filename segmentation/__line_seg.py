import math
import numpy as np
from scipy.signal import find_peaks
from __page_seg import skCorr, findScore

cwh = 27 # Correlation Window Height

def lineSegment(page):
    lineImgs = []
    page, _ = skCorr(page, 5, 0.1)
    hist = verticalCorelHist(page, cwh)
    lineCoords = findPeaks(page, hist, cwh)
    for (x, y) in lineCoords:
        imgs.append(img[x:y])
	return imgs
	
def verticalCorelHist(img, cwh = 27):
    i = 0
    lis = []
    while i < img.shape[0] - cwh:
        tem = img[i : i + cwh]
        histVal = 1 - np.sum(tem)/(math.sqrt(np.sum(np.square(tem))*cwh*img.shape[1])*255)
        lis.append(histVal)
        i += cwh
    return np.asarray(lis)
	
def findPeaks(img, corelhist, cwh):
    i = 0
    while i < corelhist.shape[0]:
        if corelhist[i] < 0.006:
            corelhist[i] = 0
        i += 1
    
    # Finding low Peaks in Correlation Histogram
    h, _ = find_peaks(1 - corelhist)
    i = h[-1] + 1
    while i < corelhist.shape[0] - 1:
        if corelhist[i] > corelhist[i-1]:
            while i<corelhist.shape[0] and corelhist[i]!=0:
                i += 1
            h = np.append(h, [i])
            break
        else:
            i += 1
    coords = []
    coords.append((0,h[0]))
    i = 0
    while i<len(h)-1:
        ele1 = h[i]
        ele2 = h[i+1]
        if (np.sum(corelhist[ele1:ele2]))<0.0155:
            i = i+1
            continue
        coords.append((ele1,ele2))
        i = i+1
    
    # From Correlation Histogram Peak Coordinates to Actual Coords
    hist = np.sum(img, axis = 0)
    hist = np.convolve(hist, np.ones((5))/5, mode = 'valid')
    for i in range(len(coords)):
        t1 = (coords[i][0] - 1) * cwh
        t2 = (coords[i][1] + 1) * cwh
        tempHist1 = hist[max(0,int(t1)):int(t1+cwh)]
        tempHist2 = hist[max(0,int(t2)):int(t2+cwh)]
        
        upperPeaks ,_ = find_peaks(-tempHist1)
        lowerPeaks ,_ = find_peaks(-tempHist2)
        i1 = upperPeaks[0] if upperPeaks.size>0 else 0
        i2 = lowerPeaks[0] if lowerPeaks.size>0 else cwh
        coords[i] = (max(0, int(t1)) + i1, max(0, int(t2)) + i2)
        
    # First Line Removal or Not
    c01, c02 = coords[0]
    temp = img[c01:c02]
    histo = np.sum(temp, axis = 0)/(c02 - c01)
    count = 0
    for j in range(histo.shape[0]):
        if histo[j] < 255 and histo[j] > 230:
            count = count+1
    if count/histo.shape[0] > 0.75:
        coords.remove((c01, c02))
    return coords