#import sys

#sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np

# Read image
imname = "Train/41.jpg"
im = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 50
params.maxThreshold = 150

params.filterByColor = False

# Filter by Area
params.filterByArea = True
params.minArea = 200
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

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save result
cv2.imwrite("result.jpg", im_with_keypoints)

# Show blobs
cv2.namedWindow(imname, cv2.WINDOW_NORMAL)
cv2.imshow(imname, im_with_keypoints)
cv2.waitKey(0)

