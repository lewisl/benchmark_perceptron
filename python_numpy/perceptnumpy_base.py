#! /usr/bin/env python3.4
# -*- encoding: utf-8 -*-


import random as r
import numpy as np

def randper(bign=10):
    cnt = 0
    disagree = 0
    runs = 1000                      # after inner loop vectorized, vectorize outer  TODO
    crossn = 10 * bign
    np.random.seed(1)
    r.seed(1)
    for k in range(runs):
        fa = (r.uniform(-1,1), r.uniform(-1,1))
        fb = (r.uniform(-1,1), r.uniform(-1,1))
        slope = (fb[1] - fa[1]) / (fb[0] - fa[0])
        intercept = fb[1] - fb[0] * slope

        # created the simulated dataset
        x = 2.0 * np.random.rand(bign, 2) - 1.0
        fx = slope * x[:, 0] + intercept
        y = np.where(x[:, 1] >= fx, 1.0, -1.0)

        # add column of ones
        x = np.column_stack((np.ones((bign, 1)), x))

        # initial pla hypothesis with weights equal 0
        w = np.array([0.0, 0.0, 0.0])
        h = x.dot(w)  # vectorized
        # print h

        # perform pla to determine g
        while True:
            misclassified = []
            for i in range(bign):
                if (y[i] < 0.0) != (h[i] < 0.0) or h[i] == 0.0:
                    misclassified.append(i)                      # VECTORIZE THIS TODO
            # print "no. misclassified", len(misclassified)
            # print misclassified
            if len(misclassified) == 0:
                break
            else:
                cnt+=1
                pick = r.randint(0,len(misclassified)-1)
                # print ("first pick", pick)
                pick = misclassified[pick]
                # print ("actual pick", pick)

                # print ("calc new weights")                       # VECTORIZE THIS TODO
                w[0] += y[pick]
                w[1] += y[pick]*x[pick][1]
                w[2] += y[pick]*x[pick][2]
                h[pick] = sum( (w[0]*1.0, w[1]*x[pick][1], w[2]*x[pick][2]) )

        # simulate a cross-validation set

        x = 2.0 * np.random.rand(crossn, 2) - 1.0
        fx = slope * x[:, 0] + intercept
        y = np.where(x[:, 1] >= fx, 1.0, -1.0)

        # x = np.zeros((crossn,2))
        # y = np.zeros((crossn))
        #
        # for i in range(crossn):
        #     x[i] = [r.uniform(-1,1), r.uniform(-1,1)]
        #     fx = slope * x[i][0] + intercept
        #     y[i] = ( 1.0 if x[i][1] >= fx  else -1.0)

        # next, add the ones
        x = np.column_stack((np.ones((crossn, 1)), x))

        # calculate hypothesis for cross validation set
        h = x.dot(w)  # vectorized

        for i in range(crossn):
            if (y[i] < 0.0) != (h[i] < 0.0) or h[i] == 0.0:
                disagree += 1                                      # VECTORIZE THIS TODO
                # print "DISAGREE", disagree
        # disagreepct += float(disagree)/float(bign)
        # print disagreepct

    print(float(cnt)/float(runs), float(disagree)/(float(runs)*float(crossn)))

    # evaluate g on a different set of points than those used to estimate g
