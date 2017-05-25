import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np
import os
import skimage.feature
import matplotlib.pyplot as plt
import imutils
import skimage

templates = ['t0.jpg', 't1.jpg', 't2.jpg', 't3.jpg', 't4.jpg',
			't5.jpg', 't6.jpg', 't7.jpg', 't8.jpg', 't9.jpg']
methods = [
    'cv2.TM_CCOEFF',
    'cv2.TM_CCOEFF_NORMED',
    'cv2.TM_CCORR']

def templateMatch(image, template):
	sealion = cv2.imread('templates/' + template)
	for angle in [0, 45, 90]:
		sealion_rot = imutils.rotate_bound(sealion, angle)
		w, h = sealion_rot.shape[1], sealion_rot.shape[0]
		sealion_noise = skimage.util.random_noise(sealion_rot, mode='gaussian')
		
		method = eval('cv2.TM_CCORR_NORMED')
		res = cv2.matchTemplate(image, sealion_rot, method)
		threshold = 0.8
		loc = np.where( res >= threshold)
		for pt in zip(*loc[::-1]):
			cv2.rectangle(image,pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
		'''
		res = cv2.matchTemplate(image, sealion_noise, method)
		threshold = 0.8
		loc = np.where( res >= threshold)
		for pt in zip(*loc[::-1]):
			cv2.rectangle(image,pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
		'''
	'''
	plt.figure(figsize=(12,8))
	plt.subplot(121)
	plt.imshow(res,cmap = 'gray')
	plt.title('Matching Result')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122)
	plt.imshow(image,cmap = 'gray')
	plt.title('Detected Point')
	plt.xticks([]), plt.yticks([])
	plt.suptitle('cv2.TM_CCOEFF_NORMED')
	plt.show()
	'''

TRAINDIR = os.path.join('TrainSmall2', 'Train')
TRAINDOTDIR = os.path.join('TrainSmall2', 'TrainDotted')
OUTPUTDIR = 'results/'

# Add values to evaluate more training images
train_ids = [41]

for i in train_ids: 
	imname_train = os.path.join(TRAINDIR, str(i) + ".jpg")
	im_train = cv2.imread(imname_train)
	im_train_gray = cv2.cvtColor(im_train, cv2.COLOR_BGR2GRAY)

	for template in templates:
		templateMatch(im_train, template)

	cv2.imwrite(OUTPUTDIR + str(i)+ ".png", im_train)