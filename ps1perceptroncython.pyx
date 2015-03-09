#! /usr/bin/env python2.7
# -*- encoding: utf-8 -*-

import random

cpdef randper(int bign=10):
    cdef:
        int cnt=0, disagree=0, runs=1000, i, crossn=1000, pick
        double pnta[2]
        double pntb[2]
        double slope, intercept, fx
        double x[100][2]  # size must match bign
        double y[100]  # size must match bign
        double h[100]  # size must match bign
        double x_cross[1000][2]  # size must match crossn
        double y_cross[1000]  # size must match crossn
        double h_cross[1000]  # size must match crossn
        double w[3]
        list misclassified

    for k in range(runs):
        pnta[0] = random.uniform(-1,1)
        pnta[1] = random.uniform(-1,1)
        pntb[0] = random.uniform(-1,1)
        pntb[1] = random.uniform(-1,1)
        slope = (pntb[1] - pnta[1]) / (pntb[0] - pnta[0])
        intercept = pntb[1] - pntb[0] * slope

        # created the simulated dataset
        for i in range(bign):
            x[i][0] =  random.uniform(-1,1)
            x[i][1] = random.uniform(-1,1)
            fx = slope * x[i][0] + intercept
            y[i] = 1.0 if x[i][1] >= fx  else -1.0
            # print x[i], fx, y[i]

        # initial pla hypothesis with weights equal 0
        w[0] = 0.0
        w[1] = 0.0
        w[2] = 0.0
        for i in range(bign):
            h[i] = sum( (w[0] * 1.0, w[1] * x[i][0], w[2] * x[i][1]) )
            # print i, y[i], h[i]
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
                pick = random.randint(0,len(misclassified)-1)
                # print "first pick", pick
                pick = misclassified[pick]
                # print "actual pick", pick

                # print "calc new weights"
                w[0] += y[pick]
                w[1] += y[pick] * x[pick][0]
                w[2] += y[pick] * x[pick][1]
                h[pick] = sum( (w[0] * 1.0, w[1] * x[pick][0], w[2] * x[pick][1]) )

        # simulate a cross-validation set
        for i in range(crossn):
            x_cross[i][0] =  random.uniform(-1,1)
            x_cross[i][1] = random.uniform(-1,1)
            fx = slope * x_cross[i][0] + intercept
            y_cross[i] = 1.0 if x_cross[i][1] >= fx  else -1.0

        # calculate hypothesis for cross validation set
        for i in range(crossn):
            h_cross[i] = sum((w[0] * 1.0, w[1] * x_cross[i][0], w[2] * x_cross[i][1]))

        # disagree = 0
        for i in range(crossn):
            if (y_cross[i] < 0) != (h_cross[i] < 0) or h_cross[i] == 0:
                disagree += 1
                # print "DISAGREE", disagree

        # disagreepct += float(disagree)/float(bign)
        # print disagreepct


    print float(cnt)/float(runs), float(disagree)/(float(runs)*float(crossn))

if __name__ == '__main__':
    randper(100)

    # evaluate g on a different set of points than those used to estimate g


