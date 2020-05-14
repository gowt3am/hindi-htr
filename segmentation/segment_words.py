import cv2, sys, argparse, os
import numpy as np
sys.path.append("../utils")
from visualize import *
from __page_seg import *
from __line_seg import *

parser = argparse.ArgumentParser('Document Segmentation Directories')
parser.add_argument('--doc', type = str, default = None)
parser.add_argument('--dir', type = str, default = "Segmented")
args = parser.parse_args()

if args.doc is None:
    print('Enter a Document Image for Segmenting')
	exit()
if os.path.exists(args.dir):
	os.rmdir(args.dir)
os.mkdir(args.dir)
os.mkdir(args.dir + '/Temp')
os.mkdir(args.dir + '/Words')

def showSaveImage(loc, img, show = False):
    if show:
	    showImage(img, 1)
	cv2.imwrite(args.dir + loc, img)

nL = 0
nW = 0
img = cv2.imread(args.doc, 0)
showSaveImage('/Temp/Document.png', img)
img = pageSegment(img)
showSaveImage('/Temp/Page.png', img)
lines = lineSegment(img, hist)
for idx, line in enumerate(lines):
    showSaveImage('/Temp/L' + str(idx + 1) + '.png', line)
	words = wordSegment(line)
	nL += 1
	for jdx, word in enurmerate(words):
	    showSaveImage('/Words/L' + str(idx + 1) + 'W' + str(jdx + 1) + '.png', word)
		nW += 1

print(str(nL) + ' Lines Segmented from Image')
print(str(nW) + ' Words Segmented from Image')