import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np
import os
import skimage.feature
import matplotlib.pyplot as plt


templates = ['brown.png', 'purple.png', 'green.png', 'red.png', 'blue.png']
methods = [
    'cv2.TM_CCOEFF',
    'cv2.TM_CCOEFF_NORMED',
    'cv2.TM_CCORR']

def templateMatch(image, template):
	sealion = cv2.imread('copy/' + template)
	w, h = sealion.shape[1], sealion.shape[0]
	method = eval('cv2.TM_CCOEFF_NORMED')
	res = cv2.matchTemplate(image, sealion, method)
	# Apply template Matching
	#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	#top_left = max_loc
	#bottom_right = (top_left[0] + w, top_left[1] + h)
	threshold = 0.8
	loc = np.where( res >= threshold)
	for pt in zip(*loc[::-1]):
		cv2.rectangle(image,pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
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
train_ids = range(41, 51)


#im = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 50
params.maxThreshold = 150

params.filterByColor = False

# Filter by Area
params.filterByArea = True
params.minArea = 0
params.maxArea = 4000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.2
params.maxCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
params.maxConvexity = 1.0

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1
params.maxInertiaRatio = 0.4

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


for i in train_ids: 
	imname_train = os.path.join(TRAINDIR, str(i) + ".jpg")
	im_train = cv2.imread(imname_train)
	im_train_gray = cv2.cvtColor(im_train, cv2.COLOR_BGR2GRAY)

	for template in templates:
		templateMatch(im_train, template)

	cv2.imwrite(OUTPUTDIR + str(i)+ ".png", im_train)