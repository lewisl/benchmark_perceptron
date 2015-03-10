#! /usr/bin/env python2.7
# -*- encoding: utf-8 -*-

import random as r
import sys

def randper(bign=10):
    cnt = 0
    disagree = 0
    runs = 1000
    r.seed(1)
    for k in range(runs):
        fa = (r.uniform(-1, 1), r.uniform(-1, 1))
        fb = (r.uniform(-1, 1), r.uniform(-1, 1))
        slope = (fb[1] - fa[1]) / (fb[0] - fa[0])
        intercept = fb[1] - fb[0] * slope

        # created the simulated dataset
        x = []
        y = []
        for i in range(bign):
            x.append(  (r.uniform(-1, 1), r.uniform(-1, 1))  )
            fx = slope * x[i][0] + intercept
            y.append( 1.0 if x[i][1] >= fx  else -1.0)
            # print x[i], fx, y[i]

        # initial pla hypothesis with weights equal 0
        w = [0.0, 0.0, 0.0]
        h = []
        for i in range(bign):
            h.append(  sum( (w[0] * 1, w[1] * x[i][0], w[2] * x[i][1]) )  )
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
                # pick = r.randint(0,len(misclassified)-1)   # randomly chosen point
                # print "first pick", pick
                pick = misclassified[len(misclassified) - 1]  # faster if not randomly chosen
                # pick = misclassified[pick]  # use randomly selected misclassified point
                # print "actual pick", pick

                # print "calc new weights"
                w[0] += y[pick]
                w[1] += y[pick] * x[pick][0]
                w[2] += y[pick] * x[pick][1]
                h[pick] = sum( (w[0] * 1, w[1] * x[pick][0], w[2] * x[pick][1]) )

        # simulate a cross-validation set
        x = []
        y = []
        crossn = 10 * bign
        for i in range(crossn):
            x.append(  (r.uniform(-1,1), r.uniform(-1,1))  )
            fx = slope * x[i][0] + intercept
            y.append(1.0 if x[i][1] >= fx else -1.0)

        # calculate hypothesis for cross validation set
        h = []
        for i in range(crossn):
            h.append(  sum( (w[0]*1.0, w[1]*x[i][0], w[2]*x[i][1]) )  )

        # disagree = 0
        for i in range(crossn):
            if (y[i] < 0) != (h[i] < 0) or h[i] == 0:
                disagree += 1
                # print "DISAGREE", disagree
        # disagreepct += float(disagree)/float(bign)
        # print disagreepct

    print (float(cnt)/float(runs), float(disagree)/(float(runs)*float(crossn)))

if __name__ == '__main__':
    try:
        randper(int(sys.argv[1]))
    except:
        randper()

    # evaluate g on a different set of points than those used to estimate g