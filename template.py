import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import threading


TRAINDIR = 'TrainSmall2/Train'
TEMPLATEDIR = 'templates'
OUTPUTDIR = 'Predicted_template'

def templateMatch(image, template, threshold):
##        sealion = cv2.imread(os.path.join(TEMPLATEDIR, template))
##        h, w = sealion.shape[:2]
##        cy, cx = h/2, w/2
##        sz = np.sqrt(cx ** 2 + cy ** 2)
##        threshold = 0.7
##        for angle in range(0,360,10):
##                M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
##                sealion_rot = cv2.warpAffine(sealion, M, (w, h))
##                sealion_crop = sealion_rot[30:130,30:130]
##                
##                method = eval('cv2.TM_CCOEFF_NORMED')
##                res = cv2.matchTemplate(image, sealion_crop, method)
##                loc = np.fliplr(np.argwhere(res >= threshold).round(-1))
##                for pt in {tuple(pt) for pt in loc}:
##                        cv2.rectangle(image, pt, (pt[0] + 100, pt[1] + 100), (0,0,255), 2)
	sealion = cv2.imread(os.path.join(TEMPLATEDIR, template))
	sz = [sealion.shape[1], sealion.shape[0]]
	h, w = image.shape[:2]
	cy, cx = h/2, w/2
	#threshold = 0.7
	for i in (1,):
                for angle in range(0,360,10):
                        A = cv2.getRotationMatrix2D((cx, cy), angle, 1)
                        B = cv2.getRotationMatrix2D((cx, cy), -angle, 1)
                        image_rot = cv2.warpAffine(image, A, (w, h))
                        
                        method = eval('cv2.TM_CCOEFF_NORMED')
                        res = cv2.matchTemplate(image_rot, sealion, method)
                        loc = np.fliplr(np.argwhere(res >= threshold))
                        loc1 = np.dot(np.insert(loc, 2, 1, 1), B.T).round(-1).astype(int)
                        loc2 = np.dot(np.insert(loc+sz, 2, 1, 1), B.T).round(-1).astype(int)
                        for (pt1, pt2) in {(tuple(pt1), tuple(pt2)) for (pt1, pt2) in zip(loc1, loc2)}:
                                cv2.rectangle(image, pt1, pt2, (0,165,255), 2)
                sealion = np.fliplr(sealion)


# Add values to evaluate more training images
train_ids = [41]
#templates = ["t1.jpg", "t0.jpg", "t2.jpg", "t3.jpg", "t6.jpg", "t7.jpg", "t9.jpg"]
#thresholds = [0.75, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
templates = ["g4.png", "g2.png", "g6.png", "g7.png", "g1.png", "g5.png"]
thresholds = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

for i in train_ids: 
        imname_train = os.path.join(TRAINDIR, str(i) + ".jpg")
        im_train = cv2.imread(imname_train)
	#threshold = 0.7 #0.75
        
        # Threaded version
        threads = [threading.Thread(target=templateMatch, args=(im_train, template, threshold)) for template, threshold in zip(templates, thresholds)]
        for thread in threads:
                thread.start()
        for thread in threads:
                thread.join()
        '''
        for template, threshold in zip(templates, thresholds):#os.listdir(TEMPLATEDIR):
                #im_train = cv2.imread(imname_train)
                templateMatch(im_train, template, threshold)
                #threshold = 0.65 #0.7
                #cv2.imwrite(os.path.join(OUTPUTDIR, template), im_train)
        '''
        cv2.imwrite(os.path.join(OUTPUTDIR, str(i) + "f.jpg"), im_train)
