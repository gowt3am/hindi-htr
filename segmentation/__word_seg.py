import cv2, math, sys
sys.path.append("../utils")
import numpy as np
from skimage import measure
from scipy.signal import find_peaks
from scipy.ndimage import interpolation as inter
from __page_seg.py import skCorr, findScore
from visualize import *

def wordSegment(line):
	chops = []
	lines = []
	skC = []
	os.chdir(src)
	i = 1	
	_, rotAng = skCorr(line[int(0.2*img.shape[0]):int(0.8*img.shape[0]), :], 3, 0.1)
	line = inter.rotate(line, rotAng, reshape = False, order = 0, cval = 255)	
	line =  movingWindowCropTopBot(line, 10, 1)
	words = returnWords(line)
	return words
	
def movingWindowCropTopBot(img, wstrip = 10, hstrip = 1):
    j = 0
    while j < img.shape[1]:
        i = 0
        it  = img[:, j:j + wstrip]
        lis = []
        while i < img.shape[0] - hstrip:
            tem = it[i:i + hstrip]
            histVal = np.sum(tem)/max(1, (math.sqrt(np.sum(np.square(tem))*hstrip*tem.shape[1])*255))
            lis.append(histVal)
            i += hstrip
        data = topBotCropper(img, np.asarray(lis), wstrip, hstrip, j)
        j += wstrip
    return data
	
def topBotCropper(img, hist, wstrip, hstrip, j, cutFrac1 = 0.15, cutFrac2 = 0.25):
    i = 0
    while i < hist.shape[0]:
        if hist[i] > 0.99:
            hist[i] = 1
        i+=1
    
	i = 0
    topCor = i
    while i < cutFrac1 * hist.shape[0]:
        if hist[i] > hist[i-1]:
            while i < cutFrac2 * hist.shape[0]:
                if hist[i] == 1:
                    topCor = i
                    break
                i+=1
            break
        i+=1
    hist[:topCor] = 1
    topCor = topCor * hstrip
    img[:topCor, j:j + wstrip] = 255
    
    i = hist.shape[0] - 1
    bottomCor = i
    while i > (1 - cutFrac1) * hist.shape[0]:
        if hist[i] > hist[i-1]:
            while i > (1 - cutFrac2) * hist.shape[0]:
                if hist[i] == 1:
                    bottomCor = i
                    break
                i-=1
            break
        i-=1
    hist[bottomCor:] = 1
    bottomCor = bottomCor * hstrip
    img[bottomCor:, j:j + wstrip] = 255
    data = img
    return data
	
def returnWords(line):
    selected, line = ccProps(line)    
    i = 0
    avgDist = 0
    while i < len(selected)-1:
        l1, Cmid1, bbox1, area1 = selected[i]
        l2, Cmid2, bbox2, area2 = selected[i+1]
        x1, y1, x2, y2 = bbox1
        X1, Y1, X2, Y2 = bbox2
        avgDist += Y1 - y2
        i += 1
    avgDist /= len(selected)
    
	i = 0
    while i < len(selected)-1:
        l1, Cmid1, bbox1, area1 = selected[i]
        l2, Cmid2, bbox2, area2 = selected[i+1]
        x1, y1, x2, y2 = bbox1
        X1, Y1, X2, Y2 = bbox2
        if abs(Y1 - y2) < avgDist*0.10 or y2 > Y1:
            newbbox = (min(x1,X1), min(y1,Y1), max(x2,X2), max(Y2,y2))
            newarea = float(area1 + area2)
            newcentroid = ((min(x1,X1) + max(x2,X2))/2, (y1+Y2)/2)
            selected.remove((l1, Cmid1, bbox1, area1))
            selected.insert(i, (l1, newcentroid, newbbox, newarea))
            selected.remove((l2, Cmid2, bbox2, area2))
        else:
            i += 1
    
    coords = []
    for sel in selected:
        x1, y1, x2, y2 = sel[2]
        if abs((y2 - y1)) > line.shape[1]*0.7:
            temCoords = SplLineCoords(line)
            coords = []
            for x, y in temCoords:
                coords.append((0, x, line.shape[0], y))
            break
        else:
            coords.append(sel[2])
     
    l3 = np.zeros(line.shape+tuple([3]))
    l3[:,:,0] = line
    l3[:,:,1] = line
    l3[:,:,2] = line
    for l1, cmid, bbox, area in selected:
        x1, y1, x2, y2 = bbox
        l3 = cv2.rectangle(l3, (y1,x1), (y2,x2), (0,255,0), 3)
    #showImage(l3, 0)
    
    wordsImgs = []
    for x1, y1, x2, y2 in coords:
        if (x2 - x1) * (y2 - y1) >= 500:
            timg = line[x1:x2, y1:y2]
            wordsImgs.append(timg)
    return wordsImgs

def ccProps(line):
    img = 255 - line
	DEkernel = np.ones((5, 2), np.uint8)
    img = cv2.dilate(img, DEkernel, iterations = 5)
    labels = measure.label(img, neighbors = 8)
    props = measure.regionprops(labels)
    num = len(props)
    
    avgArea = 0
    avgCentre = 0      #Weighted with length of BBox
    avglength = 0 
    for prop in props:
        x1, y1, x2, y2 = prop.bbox
        avglength += y2 - y1
    totlength = avglength
    avglength /= num
    
    maxdiff = 0
    center = line.shape[0]/2
    for prop in props:
        x1, y1, x2, y2 = prop.bbox
        avgArea += prop.filled_area
        if y2 - y1 > maxdiff:
            maxdiff = y2 - y1
            center = prop.centroid[0]
        avgCentre += prop.centroid[0] * (y2 - y1)/totlength
    avgArea /= num
    areaThres = 0.7*avgArea
    
    selected = []
    if maxdiff > 0.7*line.shape[1]:
        labels = measure.label(255 - line, neighbors = 8)
        _props = measure.regionprops(labels)
        l3 = np.zeros(line.shape + tuple([3]))
        l3[:,:,0] = line
        l3[:,:,1] = line
        l3[:,:,2] = line
        for _prop in _props:
            x1, y1, x2, y2 = _prop.bbox
            l3 = cv2.rectangle(l3, (y1, x1), (y2, x2), (0, 255, 0), 5)
            RMID = _prop.centroid[0]
            if abs(RMID - center) >= 0.15 * line.shape[0]:    
                l3 = cv2.rectangle(l3, (y1, x1), (y2, x2), (255, 0, 0), 5)
                line[x1:x2, y1:y2] = 255
        for prop in props:
            selected.append((prop.label, prop.centroid, prop.bbox, prop.filled_area))
    else:
        for prop in props:
            x1, y1, x2, y2 = prop.bbox
            Rmid = prop.centroid[0]
            if abs(Rmid - avgCentre) > 0.22*line.shape[0] and prop.filled_area<areaThres: 
                line[x1:x2, y1:y2] = 255
            else:
                selected.append((prop.label, prop.centroid, prop.bbox, prop.filled_area))
            
    selected = sorted(selected, key = lambda x : x[1][1])
    return selected, line

def corelHistW(line, wstrip = 50):
    i = 0
    lis = []
    while i < line.shape[1] - wstrip:
        tem = line[:, i:i + wstrip]
        histVal = 1 - np.sum(tem)/(math.sqrt(np.sum(np.square(tem))*wstrip*line.shape[0])*255)
        lis.append(histVal)
        i += wstrip
    return np.asarray(lis)
	
def SplLineCoords(line, wstrip = 30, wstripAvg = 50):
    #Special Function to tackle lines with shirorekha touching header line
	#Ignore for other documents
    avgchist = corelHistW(line, wstripAvg)
    i = 0
    while i < avgchist.shape[0]:
        if avgchist[i] < 0.006:        #Threshold to avoid noisy peaks
            avgchist[i] = 0
        i += 1
    temh,_ = find_peaks(1 - avgchist)
    
    thresh = 0
    for i in temh:
        thresh += avgchist[i]
    thresh /= len(temh)
    thresh = 1.05*thresh if thresh < 0.4 else thresh
    
    chist = corelHistW(line, wstrip)
    i = 0
    while i < chist.shape[0]:
        if chist[i] < 0.006:        #Threshold to avoid noisy peaks
            chist[i] = 0
        i += 1
    h,_ = find_peaks(1 - chist, height = 1 - thresh)
    
    i = h[-1] + 1
    while i < chist.shape[0]-1:
        if chist[i] > chist[i-1]:
            while i < chist.shape[0] and chist[i] != 0:
                i += 1
            h = np.append(h, [min(i+5, line.shape[1])])
            break
        else:
            i += 1
            
    coords = []
    coords.append((0, h[0]))
    i = 0
    while i < len(h) - 1:
        ele1 = h[i]
        ele2 = h[i + 1]
        if (np.sum(chist[ele1:ele2])/(ele2-ele1)) < 0.017:   #Threshold to avoid complete white regions
            i = i + 1
            continue
        coords.append((ele1, ele2))
        i = i + 1
        
    for i in range(len(coords)):
        t1 = (coords[i][0]) * wstrip
        t2 = (coords[i][1]) * wstrip
        coords[i] = (max(0, int(t1)), max(0, int(t2 + wstrip/2)))
    return coords