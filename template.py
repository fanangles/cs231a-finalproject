import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

TRAINDIR = 'Train'
TEMPLATEDIR = 'templates'
OUTPUTDIR = 'Predicted_template'

def non_max_suppression(points, confidences, size):
    if len(points) == 0:
        return points
    sortpoints = points[np.argsort(confidences)]
    nms_points = [sortpoints[0] + size / 2]
    for center in sortpoints + size / 2:
        if not ((center >= nms_points - size / 2) &
                (center <= nms_points + size / 2)).all(1).any():
            nms_points.append(center)
    return np.array(nms_points)

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
    kernel = np.ones((3,3), np.uint8)
    sealion = cv2.imread(os.path.join(TEMPLATEDIR, template))
    size = np.array([sealion.shape[1], sealion.shape[0]])
    h, w = image.shape[:2]
    cy, cx = h/2, w/2
    for i in (1,):
        for angle in range(0,360,10):
            A = cv2.getRotationMatrix2D((cx, cy), angle, 1)
            B = cv2.getRotationMatrix2D((cx, cy), -angle, 1)
            image_rot = cv2.warpAffine(image, A, (w, h))
            res = cv2.matchTemplate(image_rot, sealion, cv2.TM_CCOEFF_NORMED)
            opn = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
            loc = np.fliplr(np.argwhere(opn >= threshold))
            nms_loc = non_max_suppression(loc, opn[opn >= threshold], size)
            pts = np.dot(np.insert(nms_loc, 2, 1, 1), B.T)
            for pt in pts:
                box = cv2.boxPoints((pt, size, angle))
                cv2.drawContours(image,[np.int32(box)],0,(0,165,255),2)
        sealion = np.fliplr(sealion)


# Add values to evaluate more training images
train_ids = [48]
#templates = ["t1.jpg", "t0.jpg", "t2.jpg", "t3.jpg", "t6.jpg", "t7.jpg", "t9.jpg"]
#thresholds = [0.74, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
templates = ["f4.jpg", "f2.jpg", "f6.jpg", "f7.jpg", "f1.jpg", "f5.jpg"]
thresholds = [0.7, 0.5, 0.67, 0.65, 0.5, 0.6]
#templates = ["j0.jpg", "j1.jpg", "j3.jpg", "j4.jpg", "j5.jpg", "j7.jpg", "j8.jpg"]
#thresholds = [0.67, 0.5, 0.8, 0.65, 0.73, 0.75, 0.72]

for i in train_ids: 
    imname_train = os.path.join(TRAINDIR, str(i) + ".jpg")
    im_train = cv2.imread(imname_train)
    
    threads = [threading.Thread(target=templateMatch, args=(im_train, template, threshold)) for template, threshold in zip(templates, thresholds)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

##    for template, threshold in zip(templates, thresholds):#os.listdir(TEMPLATEDIR):
##        #im_train = cv2.imread(imname_train)
##        templateMatch(im_train, template, threshold)
##        #cv2.imwrite(os.path.join(OUTPUTDIR, template), im_train)

    cv2.imwrite(os.path.join(OUTPUTDIR, str(i) + ".jpg"), im_train)
