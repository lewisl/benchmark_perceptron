#! /usr/bin/env python2.7
# -*- encoding: utf-8 -*-



import random as r
import numpy as np


def randper(bign):
    cnt = 0
    disagree = 0
    runs = 1000

    # arrays for the simulated dataset
    x = np.zeros((bign, 2))
    # add column of ones
    x = np.column_stack((np.ones((bign, 1)), x))
    y = np.zeros((bign))

    # arrays for the simulated cross-validation dataset
    crossn = 10 * bign
    x_cross = np.zeros((crossn, 2))
    y_cross = np.zeros((crossn))
    # next, add the ones
    x_cross = np.column_stack((np.ones((crossn, 1)), x_cross))

    # initial pla hypothesis with weights equal 0
    w = np.array([0.0, 0.0, 0.0])

    for k in range(runs):
        fa = (r.uniform(-1, 1), r.uniform(-1, 1))
        fb = (r.uniform(-1, 1), r.uniform(-1, 1))
        slope = (fb[1] - fa[1]) / (fb[0] - fa[0])
        intercept = fb[1] - fb[0] * slope

        # created the simulated dataset

        for i in range(bign):
            x[i][1:] = (r.uniform(-1, 1), r.uniform(-1, 1))
            fx = slope * x[i][1] + intercept
            y[i] = (1 if x[i][2] >= fx else -1)
            # print x[i], fx, y[i]

        # Calculate PLA hypothesis values
        # for i in range(bign):
        #     h[i] = (  sum( (w[0]*1, w[1]*x[i][0], w[2]*x[i][1]) )  )
        #     # print i, y[i], h[i]
        h = x.dot(w)  # vectorized
        # print h

        # perform pla to determine g
        while True:
            misclassified = []
            for i in range(bign):
                if (y[i] < 0) != (h[i] < 0) or h[i] == 0:
                    misclassified.append(i)
            # print "no. misclassified", len(misclassified)
            # print misclassified
            if len(misclassified) == 0:
                break
            else:
                cnt += 1
                pick = r.randint(0, len(misclassified) - 1)
                # print "first pick", pick
                pick = misclassified[pick]
                # print "actual pick", pick

                # print "calc new weights"
                w[0] += y[pick]
                w[1] += y[pick] * x[pick][1]
                w[2] += y[pick] * x[pick][2]
                h[pick] = sum((w[0] * 1.0, w[1] * x[pick][1], w[2] * x[pick][2]))

        # simulate a cross-validation set
        for i in range(crossn):
            x_cross[i][1:] = (r.uniform(-1, 1), r.uniform(-1, 1))
            fx = slope * x_cross[i][1] + intercept
            y_cross[i] = (1.0 if x_cross[i][2] >= fx else -1.0)

        # calculate hypothesis for cross validation set
        # hcheck = np.zeros((crossn))
        # for i in range(crossn):
        #     hcheck[i] = (  sum( (w[0]*1, w[1]*x[i][0], w[2]*x[i][1]) )  )
        h_cross = x_cross.dot(w)  # vectorized

        for i in range(crossn):
            if (y_cross[i] < 0) != (h_cross[i] < 0) or h_cross[i] == 0:
                disagree += 1
                # print "DISAGREE", disagree
                # disagreepct += float(disagree)/float(bign)
                # print disagreepct

    print float(cnt) / float(runs), float(disagree) / (float(runs) * float(crossn))

    # evaluate g on a different set of points than those used to estimate g
