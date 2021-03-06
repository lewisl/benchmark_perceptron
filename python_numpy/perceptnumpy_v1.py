#! /usr/bin/env python3.4
# -*- encoding: utf-8 -*-

import numpy as np


def randper(bign):
    cnt = 0
    disagree = 0
    runs = 1000
    crossn = 10 * bign
    np.random.seed(1)

    for k in range(runs):
        fa = np.random.uniform(-1, 1, 2)  # (r.uniform(-1, 1), r.uniform(-1, 1))
        fb = np.random.uniform(-1, 1, 2)  # (r.uniform(-1, 1), r.uniform(-1, 1))
        slope = (fb[1] - fa[1]) / (fb[0] - fa[0])
        intercept = fb[1] - fb[0] * slope

        # print("slope: \n", slope)
        # print("intercept: \n", intercept)

        # create the simulated dataset
        x = 2.0 * np.random.rand(bign, 2) - 1.0
        fx = slope * x[:, 0] + intercept          # this is slow: scalar ops with vector
        y = np.where(x[:, 1] >= fx, 1.0, -1.0)

        # print("fx: ", fx)
        # print("y: ", y)
        # add column of ones
        x = np.concatenate((np.ones((bign, 1)), x), axis=1)
        # print("first 5 runs for x\n",x[0:5])

        # Calculate PLA hypothesis values
        # initial pla hypothesis with weights equal 0
        w = np.array([0.0, 0.0, 0.0])
        h = x.dot(w)  # vectorized
        # print("h: ", h)

        # perform pla to determine g
        while True:
            misclassified = np.argwhere((np.where(y < 0.0, 1, -1) * np.where(h < 0.0, 1, -1)
                                         + np.where(h == 0.0, -1, 0)) <= 0)
            if len(misclassified) == 0:
                break
            else:
                cnt += 1
                # pick = r.randint(0, len(misclassified) - 1)
                pick = misclassified[len(misclassified) - 1][0]

                # calc new weights and new hypothesis value
                w[0] += y[pick]
                w[1] += y[pick] * x[pick][1]
                w[2] += y[pick] * x[pick][2]
                h[pick] = x[pick].dot(w)

        # print("w: ", w)
        # simulate a cross-validation set
        # evaluate g on a different set of points than those used to estimate g
        x_cross = 2.0 * np.random.rand(crossn, 2) - 1.0
        fx = slope * x_cross[:, 0] + intercept
        y_cross = np.where(x_cross[:, 1] >= fx, 1, -1)
        # next, add the ones
        x_cross = np.concatenate((np.ones((crossn, 1)), x_cross), axis=1)
        h_cross = x_cross.dot(w)

        yless0 = np.where(y_cross < 0.0, 1, 0)
        hless0 = np.where(h_cross < 0.0, 1, 0)
        heq0 = np.where(h_cross == 0.0, 1, 0)
        ne = np.where(yless0 != hless0, 1, 0)
        disagree += np.sum(np.where(heq0 + ne > 0, 1, 0))

    # print("cnt: ", cnt, " disagree: ", disagree)
    print(float(cnt) / float(runs), disagree / (float(runs) * float(crossn)))
