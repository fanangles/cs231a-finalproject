import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

win = 72
step = 18

coordfile = "correct_coordinates.csv"
TRAINDIR = 'Train'
TEMPLATEDIR = 'templates'
OUTPUTDIR = 'Predicted_template'

def non_max_suppression(points, confidences, size):
    if len(points) == 0:
        return points
    sortpoints = points[np.argsort(confidences)[::-1]]
    nms_points = [sortpoints[0] + size / 2]
    for center in sortpoints + size / 2:
        if not ((center >= nms_points - size / 2) &
                (center <= nms_points + size / 2)).all(1).any():
            nms_points.append(center)
    return np.array(nms_points)

def train_data(imdir, coords):
    cellslist = []
    labelslist = []
    for imname in os.listdir(imdir):
        im = cv2.imread(os.path.join(imdir, imname))
        coord = coords[coords[:,0] == int(os.path.splitext(imname)[0])][:,1:]
        coord = np.maximum(0, np.minimum([im.shape[0] - win, im.shape[1] - win], coord - win/2))
        pos_cells = [np.reshape(im[y:y+win,x:x+win], (-1,)) for (y,x) in coord]
        neg_cells = [np.reshape(im[y:y+win,x:x+win], (-1,))
                     for y in xrange(0, im.shape[0] - win, win)
                     for x in xrange(0, im.shape[1] - win, win)
                     if not np.any(np.all(np.logical_and(
                         [y,x] >= coord - win/2,
                         [y,x] <= coord + win/2), 1))]
        labels = np.zeros(2000)
        labels[:len(pos_cells)] = 1
        cellslist.append(pos_cells+neg_cells[:2000-len(pos_cells)])
        labelslist.append(np.int32(labels))
    return np.concatenate(cellslist[:2]), np.concatenate(labelslist[:2])

def templateMatch(image, template, threshold, clf):
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
                score = clf.decision_function([image[pt[0]-win/2:pt[0]+win/2,
                                                     pt[1]-win/2:pt[1]+win/2]])
                if score < 0.75: continue
                box = cv2.boxPoints((pt, size, angle))
                cv2.drawContours(image,[np.int32(box)],0,(255,0,0),2)
        sealion = np.fliplr(sealion)


# Add values to evaluate more training images
train_ids = [48]
#templates = ["t1.jpg", "t0.jpg", "t2.jpg", "t3.jpg", "t6.jpg", "t7.jpg", "t9.jpg"]
#thresholds = [0.74, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
templates = ["f4.jpg", "f2.jpg", "f6.jpg", "f7.jpg", "f1.jpg", "f5.jpg"]
thresholds = [0.7, 0.5, 0.67, 0.65, 0.5, 0.6]
#templates = ["j0.jpg", "j1.jpg", "j3.jpg", "j4.jpg", "j5.jpg", "j7.jpg", "j8.jpg"]
#thresholds = [0.67, 0.5, 0.8, 0.65, 0.73, 0.75, 0.72]

coords = np.loadtxt(coordfile, int, delimiter=',', skiprows=1, usecols=(1,2,3))
cellslist, labelslist = train_data(TRAINDIR, coords)
clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
clf.fit(cellslist, labelslist)

for i in train_ids: 
    imname_train = os.path.join(TRAINDIR, str(i) + ".jpg")
    im_train = cv2.imread(imname_train)
    
    for template, threshold in zip(templates, thresholds):#os.listdir(TEMPLATEDIR):
        #im_train = cv2.imread(imname_train)
        templateMatch(im_train, template, threshold, clf)
        #cv2.imwrite(os.path.join(OUTPUTDIR, template), im_train)

    cv2.imwrite(os.path.join(OUTPUTDIR, str(i) + ".jpg"), im_train)
