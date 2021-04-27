#! /usr/bin/env python3.4
# -*- encoding: utf-8 -*-

import random as r
# import numpy as np  # np approach was too slow


def randper(bign=10):
    global crossn
    cnt = 0
    disagree = 0
    points = 1000
    r.seed(1)
    for k in range(points):
        fa = (r.uniform(-1, 1), r.uniform(-1, 1))
        fb = (r.uniform(-1, 1), r.uniform(-1, 1))
        slope = (fb[1] - fa[1]) / (fb[0] - fa[0])
        intercept = fb[1] - fb[0] * slope

        # created the simulated dataset
        # x = np.random.uniform(-1.0, 1.0, (bign,2))  # 3 times slower than looping
        x = []
        y = []
        for i in range(bign):
            x.append((r.uniform(-1, 1), r.uniform(-1, 1)))
            fx = slope * x[i][0] + intercept
            y.append(1.0 if x[i][1] >= fx else -1.0)
            # print x[i], fx, y[i]

        # initial pla hypothesis with weights equal 0
        w = [0.0, 0.0, 0.0]
        h = []
        for i in range(bign):
            h.append(sum((w[0] * 1, w[1] * x[i][0], w[2] * x[i][1])))
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
                pick = misclassified[len(misclassified) - 1]  # faster than random pick
                # pick = misclassified[pick]  # use randomly selected misclassified point
                # print "actual pick", pick

                # print "calc new weights"
                w[0] += y[pick]
                w[1] += y[pick] * x[pick][0]
                w[2] += y[pick] * x[pick][1]
                h[pick] = sum((w[0] * 1, w[1] * x[pick][0], w[2] * x[pick][1]))

        # simulate a cross-validation set
        x = []
        y = []
        h = []
        crossn = 10 * bign
        for i in range(crossn):
            x.append((r.uniform(-1, 1), r.uniform(-1, 1)))
            fx = slope * x[i][0] + intercept
            y.append(1.0 if x[i][1] >= fx else -1.0)
            # calculate hypothesis for cross validation set
            h.append(sum((w[0] * 1.0, w[1] * x[i][0], w[2] * x[i][1])))
            if (y[i] < 0) != (h[i] < 0) or h[i] == 0:
                disagree += 1

    print(float(cnt) / float(points),
          float(disagree) / (float(points) * float(crossn)))

if __name__ == '__main__':
    randper(100)
