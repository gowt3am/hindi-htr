import os, sys, random
sys.path.append("../utils")
import numpy as np

def charCombinations(wordRepo, charFolders, copies):
    wordRep = open(wordRepo, 'r', encoding = 'utf-8')
    conjunct = ['091C_094D', '0915_094D', '0924_094D']

	wordComb = []
	for word in wordRep:
		charList = []
		if word[0] != '#' and len(word) <= 31:
			word = word.replace('\n', '')
			characters = []
			for ch in word:
				characters.append(('0' + hex(ord(ch))[2:]).upper())
			
			check = True
			i = 0
			while check and i < len(characters):
				check = False
				word = ''
				if i < len(characters) - 1:
					word = characters[i] + '_' + characters[i+1]
				
                if word in conjunct and i < len(characters) - 2:
					word2 = word + '_' + characters[i+2]
					if word2 in charFolders:
						if i < len(characters) - 3 and word2 + '_' + characters[i+3] in charFolders:
							charList.append(word2 + '_' + characters[i+3])
							check = True
							i += 4
						else:
							charList.append(word2)
							check = True
							i += 3
				
				if check == False and word in charFolders:
					check = True
					charList.append(word)
					i += 2
				
				if check == False and characters[i] in charFolders:
					check = True
					charList.append(characters[i])
					i += 1
			
			if check == True:
				wordComb.append((charList, word))
	wordRep.close()
	print('Num. of Unique Possible Words: ', len(wordComb))
	
	dst = os.getcwd() + '/ProcessedCALAM/'
	charImgs_Word = []
	for charList, gt in wordComb:
		for i in range(copies):
		    charImgs = []
			for char in charList:
				dir = dst + char + '/'
				file = os.path.join(dir, random.choice(os.listdir(dir)))
				charImgs.append(file)
			charImgs_Word.append((charImgs, gt))
	
	return charImgs_Word