import os, glob, cv2, sys, shutil
import numpy as np
sys.path.append("../segmentation")
from __page_seg import gaussBlur, binarize

upHeight = 110
lowHeight = 100
height = 70
upString = ['908', '910', '913', '914', '93F', '940', '947', '948', '94B', '94C']
lowString = ['941', '942']

def preprocessCALAM(src, DEkernel_size = (2,2), blur_rad = 3, resize = True,
					max_ang = 2.1, delta = 0.05, blur = True, binary = 'THRESH_OTSU'):
    cwd = os.getcwd()
	os.chdir(src)
    folderNames = []
    for tem in glob.glob('*'):
        if os.path.isdir(tem):
            folderNames.append(tem)
            
    for tem in folderNames:
        if tem[-3:] in upString:
            h = upHeight
        elif tem[-3:] in lowString:
            h = lowHeight
        else:
            h = height
			
        dst = cwd + '/ProcessedCALAM/' + tem
        if os.path.exists(dst):
		    shutil.rmtree(dst)
        os.mkdir(dst)
        
		os.chdir(src + '/' + tem + '/')
        for file in glob.glob('*.png'):
            img = cv2.imread(file, 0)
            if blur == True:
                img = gaussBlur(img, blur_rad)
            img = 255 * (1 - binarize(img, binary))
            if resize == True:
                th = img.shape[0]
                tw = img.shape[1]
                f = th/h
                newSize = (int(tw/f), h)
                img = cv2.resize(img, newSize)
            cv2.imwrite(dst + '/' + file, img)
	print(str(len(folderNames)) + ' different characters identified in CALAM')
	os.chdir(cwd)
	return folderNames