#import sys

#sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

sz = 20
nbins = 9
win = 72
step = 18

coordfile = "correct_coordinates.csv"

def deskew(image):
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5*sz*skew], [0, 1, 0]])
    img = cv2.warpAffine(image, M, (sz,sz),
                         flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img

def hog(image):
    hog = cv2.HOGDescriptor((win,win), (12,12), (6,6), (6,6), nbins)
    return hog.compute(image).flatten()

def train_data(imdir, coords):
    cellslist = []
    labelslist = []
    for imname in os.listdir(imdir):
        im = cv2.imread(os.path.join(imdir, imname), cv2.IMREAD_GRAYSCALE)
        coord = coords[coords[:,0] == int(os.path.splitext(imname)[0])][:,1:]
        pos_cells = [im[y:y+win,x:x+win] for (y,x) in coord - win/2]
        neg_cells = [im[y:y+win,x:x+win]
                     for y in xrange(0, im.shape[0] - win, win)
                     for x in xrange(0, im.shape[1] - win, win)
                     if not np.any(np.all(np.logical_and(
                         [y,x] >= coord - win/2,
                         [y,x] <= coord + win/2), 1))]
        labels = np.zeros(2000)
        labels[:len(pos_cells)] = 1
        cellslist.append(pos_cells+neg_cells[:2000-len(pos_cells)])
        labelslist.append(np.int32(labels))
    return cellslist[:2], labelslist[:2]

def classify(imdir, coords, clf):
    correct, true, pred = 0, 0, 0
    for imname in os.listdir(imdir):
        im = cv2.imread(os.path.join(imdir, imname), cv2.IMREAD_GRAYSCALE)
        coord = coords[coords[:,0] == int(os.path.splitext(imname)[0])][:,1:]
        points = np.array(
            [[y,x] for y in xrange(0, im.shape[0], step)
             for x in xrange(0, im.shape[1], step)
             if y+win <= im.shape[0] and x+win <= im.shape[1]])
        rows = (im.shape[0] - win) / step + 1
        cols = (im.shape[1] - win) / step + 1
        labels = np.zeros(rows * cols)
        labels[np.minimum(rows - 1, np.maximum(0, (coord[:,0] - win/2) / step)) * cols +
               np.minimum(cols - 1, np.maximum(0, (coord[:,1] - win/2) / step))] = 1
        scores = []
        for (y,x) in points:
            hogdata = map(hog, [im[y:y+win,x:x+win]])
            data = np.float32(hogdata)
            score = clf.decision_function(data)
            scores.append(score)
        scores = np.concatenate(scores)
        predictions = non_max_suppression(points, scores) * (scores > 1)
        draw_predictions(imdir, imname, predictions)
        correct += np.count_nonzero(predictions * labels)
        true += np.count_nonzero(labels)
        pred += np.count_nonzero(predictions)
    # recall
    print correct * 100.0 / true
    # precision
    print correct * 100.0 / pred

def draw_predictions(imdir, imname, predictions):
        im = cv2.imread(os.path.join(imdir, imname))
        cols = (im.shape[1] - win) / step + 1
        for (i, pred) in enumerate(predictions):
            if pred == 1:
                x = (i % cols) * step
                y = (i / cols) * step
                cv2.rectangle(im, (x, y), (x+win, y+win), (0, 0, 255), 2)
        cv2.imwrite(os.path.join("Predicted_hog", imname), im)

def non_max_suppression(points, confidences):
    sortpoints = np.argsort(confidences)[::-1]
    nms = np.zeros_like(sortpoints, dtype=bool)
    for i in sortpoints:
        center = points[i] + win / 2
        nms[i] = ((center < points[nms]) | (center > points[nms] + win)).any(1).all()
    return nms

coords = np.loadtxt(coordfile, int, delimiter=',', skiprows=1, usecols=(1,2,3))

# Training
cellslist, labelslist = train_data("Train", coords)
hogdata = map(hog, itertools.chain(*cellslist))
data = np.float32(hogdata)

clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
clf.fit(data, np.concatenate(labelslist))

# Testing
classify("Test", coords, clf)

