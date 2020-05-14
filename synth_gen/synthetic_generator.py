import sys, argparse
import numpy as np
sys.path.append("../segmentation")
from preprocessCALAM import *
from charCombinations import *
from generateWordDataset import *

parser = argparse.ArgumentParser('Synthetic Dataset Generation Settings')
parser.add_argument('--src', type = str, default = None)                 #Points to SegregatedCALAM folder
parser.add_argument('--dst', type = str, default = "synth1")
parser.add_argument('--copies', type = int, default = 1)
parser.add_argument('--repo', type = str, default = 'IIITH+IWN_Words.txt')
args = parser.parse_args()

copies = args.copies
if args.src is None:
    print('Enter the source directory of SegregatedCALAM folder')
	exit()
src = args.src
dst = args.dst

charFolders = preProcessCALAM(src)
charImgs_Word = charCombinations(args.repo, charFolders, args.copies)
generateWordDataset(charImgs_Word, dst)