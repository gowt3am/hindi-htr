import os, shutil, cv2
import numpy as np

upHeight = 110
lowHeight = 100
height = 70

def generateWordDataset(charImgs_Word, dst)
    cwd = os.getcwd()
	folder = cwd + '/' + dst
	file = cwd + '/' + dst + '.txt'
	if os.path.exists(folder):
		shutil.rmtree(folder)
	os.mkdir(folder)
	labelFile = open(file, 'w+', encoding = 'utf-8')
	
	count = 0
	for tup in charImgs_Word:
		count += 1
		imgs = [cv2.imread(file, 0) for file in tup[0]]
		if len(imgs) == 0:
			continue
			
		h = height
		topPresent = False
		botPresent = False
		width = 0
		for i in range(len(imgs)):
			if imgs[i].shape[0] == upHeight:
				topPresent = True
			elif imgs[i].shape[0] == lowHeight:
				botPresent = True
			imgs[i] = np.uint64(1 - imgs[i]/255)
			width += imgs[i].shape[1]*0.85
		if len(imgs) >= 2:
			width += imgs[0].shape[1]*0.15 + imgs[-1].shape[1]*0.15
		else:
			width += imgs[0].shape[1]*0.15
		width = int(width) + 1
	   
		if not topPresent and not botPresent:
			result = np.uint64(np.zeros((h, width)))
			result[:, 0:imgs[0].shape[1]] = np.bitwise_or(result[:, 0:imgs[0].shape[1]], imgs[0])
			i = 1
			index = int(0.85*imgs[i-1].shape[1])
			while i < len(imgs):
				width = imgs[i].shape[1]
				result[:, index:index+width] = np.bitwise_or(result[:, index:index+width], imgs[i])
				index += int(0.85*imgs[i].shape[1])
				i += 1
				
		elif topPresent and not botPresent:
			result = np.uint64(np.zeros((upHeight, width)))
			if imgs[0].shape[0] == upHeight:
				result[:, 0:imgs[0].shape[1]] = np.bitwise_or(result[:, 0:imgs[0].shape[1]], imgs[0])
			else:
				result[-height:, 0:imgs[0].shape[1]] = np.bitwise_or(result[-height:, 0:imgs[0].shape[1]], imgs[0])
			i = 1
			index = int(0.85*imgs[i-1].shape[1])
			while i < len(imgs):
				width = imgs[i].shape[1]
				if imgs[i].shape[0] == upHeight:
					result[:, index:index+width] = np.bitwise_or(result[:, index:index+width],imgs[i])
				else:
					result[-height:, index:index+width] = np.bitwise_or(result[-height:, index:index+width], imgs[i])
				index += int(0.85*imgs[i].shape[1])
				i += 1
				
		elif not topPresent and botPresent:
			result = np.uint64(np.zeros((lowHeight, width)))
			if imgs[0].shape[0] == lowHeight:
				result[:, 0:imgs[0].shape[1]] = np.bitwise_or(result[:, 0:imgs[0].shape[1]], imgs[0])
			else:
				result[:height, 0:imgs[0].shape[1]] = np.bitwise_or(result[:height, 0:imgs[0].shape[1]], imgs[0])
			i = 1
			index = int(0.85*imgs[i-1].shape[1])
			while i < len(imgs):
				width = imgs[i].shape[1]
				if imgs[i].shape[0] == lowHeight:
					result[:, index:index+width] = np.bitwise_or(result[:, index:index+width], imgs[i])
				else:
					result[:height, index:index+width] = np.bitwise_or(result[:height, index:index+width], imgs[i])
				index += int(0.85*imgs[i].shape[1])
				i += 1
		
		else:
			result = np.uint64(np.zeros((upHeight + lowHeight - height, width)))
			if imgs[0].shape[0] == upHeight:
				result[:upHeight, 0:imgs[0].shape[1]] = np.bitwise_or(result[:upHeight, 0:imgs[0].shape[1]], imgs[0])
			elif imgs[0].shape[0] == lowHeight:
				result[-lowHeight:, 0:imgs[0].shape[1]] = np.bitwise_or(result[-lowHeight:, 0:imgs[0].shape[1]], imgs[0])
			else:
				result[upHeight-height:upHeight, 0:imgs[0].shape[1]] = np.bitwise_or(result[upHeight-height:upHeight, 0:imgs[0].shape[1]], imgs[0])
			i = 1
			index = int(0.85*imgs[i-1].shape[1])
			while i < len(imgs):
				width = imgs[i].shape[1]
				if imgs[i].shape[0] == upHeight:
					result[:upHeight, index:index+width] = np.bitwise_or(result[:upHeight, index:index+width], imgs[i])
				elif imgs[i].shape[0] == lowHeight:
					result[-lowHeight:, index:index+width] = np.bitwise_or(result[-lowHeight:, index:index+width], imgs[i])
				else:
					result[upHeight-height:upHeight, index:index+width] = np.bitwise_or(result[upHeight-height:upHeight, index:index+width], imgs[i])
				index += int(0.85*imgs[i].shape[1])
				i += 1
		
		result = 255 - result*255
		cv2.imwrite(folder + '/' + str(count) + '.png', result)
		labelFile.write(folder + '/' + str(count) + '.png '+ str(tup[1])+ '\n')
	labelFile.close()
	
	print(str(count) + ' Synthetic Images Generated and stored at ' + folder)
	print('LabelFile is stored at ' + file)