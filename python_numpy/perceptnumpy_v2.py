#! /usr/bin/env python3.4
# -*- encoding: utf-8 -*-

import numpy as np


def randper(bign):
    cnt = 0
    # disagree = 0  # dont' need to initialize as not updated in loop in this version
    runs = 1000
    crossn = 10 * bign
    np.random.seed(1)

    fa = np.random.uniform(-1, 1, (runs, 2))
    fb = np.random.uniform(-1, 1, (runs, 2))
    slope = (fb[:, 1] - fa[:, 1]) / (fb[:, 0] - fa[:, 0])   # runs
    intercept = fb[:, 1] - fb[:, 0] * slope       # runs

    # print("slope: \n", slope)
    # print("intercept: \n", intercept)

    # print('{0:18} {1}'.format("fa shape: ", fa.shape))
    # print('{0:18} {1}'.format("fb shape: ", fb.shape))
    # print('{0:18} {1}'.format("slope shape: ", slope.shape))
    # print('{0:18} {1}'.format("intercept shape: ", intercept.shape))

    # create the simulated dataset
    x = 2.0 * np.random.rand(runs, bign, 2) - 1.0     # runs,bign,2
    fx = (np.repeat(slope[:, np.newaxis], bign, 1) * x[:, :, 0]
          + np.repeat(intercept[:, np.newaxis], bign, 1))          # runs, bign
    y = np.where(x[:, :, 1] >= fx, 1.0, -1.0)    # runs, bign
    #
    # print("fx: ", fx)
    # print("y: ", y)
    # print('{0:18} {1}'.format("x shape: ", x.shape))
    # print('{0:18} {1}'.format("fx shape: ", fx.shape))
    # print('{0:18} {1}'.format("y shape: ", y.shape))

    # add column of ones
    x = np.concatenate((np.ones((runs, bign, 1)), x), axis=2)  # runs, bign, 3
    # print('{0:18} {1}'.format("new x shape: ", x.shape))
    # print("first 5 runs for x\n",x[0:5])

    # Calculate PLA hypothesis values
    # initial pla hypothesis with weights equal 0
    w = np.zeros((runs, 3))  # need a different w per run
    h = np.einsum("ijk, ik -> ij", x, w)  # runs, bign 3  ;  runs, 3  -> runs, bign

    # print("h: ", h)
    # print('{0:18} {1}'.format("h shape: ", h.shape))
    #
    # perform pla to determine g
    for k in range(runs):
        while True:
            misclassified = np.argwhere((np.where(y[k] < 0.0, 1, -1)   # runs by variable
                * np.where(h[k] < 0.0, 1, -1) + np.where(h[k] == 0.0, -1, 0)) <= 0)
            if len(misclassified) == 0:
                break
            else:
                cnt += 1
                # pick = r.randint(0, len(misclassified) - 1)  # randomized choise of misc. point
                # pick = misclassified[pick][0]                # ditto
                pick = misclassified[len(misclassified) - 1][0]

                # calc new weights and new hypothesis value
                w[k, 0] += y[k, pick]
                w[k, 1] += y[k, pick] * x[k, pick][1]
                w[k, 2] += y[k, pick] * x[k, pick][2]
                h[k][pick] = x[k][pick].dot(w[k])

    # print("w: ", w)
    # print('{0:18} {1}'.format("w shape: ", w.shape))

    # simulate a cross-validation set  -- set up matrices as above
    # evaluate g on a different set of points than those used to estimate g
    x_cross = 2.0 * np.random.rand(runs, crossn, 2) - 1.0  # runs, crossn, 2
    fx = (np.repeat(slope[:, np.newaxis], crossn, 1) * x_cross[:, :, 0]
          + np.repeat(intercept[:, np.newaxis], crossn, 1))          # runs, crossn
    y_cross = np.where(x_cross[:, :, 1] >= fx, 1, -1)    # runs, crossn
    # # next, add the ones
    x_cross = np.concatenate((np.ones((runs, crossn, 1)), x_cross), axis=2)  # runs, crossn, 3
    # print("\nfirst 5 runs for x_cross\n",x[0:5])

    h_cross = np.einsum("ijk, ik -> ij", x_cross, w)   # runs, crossn

    # print('{0:18} {1}'.format("x_cross shape: ", x_cross.shape))
    # print('{0:18} {1}'.format("fx shape: ", fx.shape))
    # print('{0:18} {1}'.format("y_cross shape: ", y_cross.shape))
    # print('{0:18} {1}'.format("x_cross shape: ", x_cross.shape))
    # print('{0:18} {1}'.format("h_cross shape: ", h_cross.shape))

    yless0 = np.where(y_cross < 0, 1, 0)      # runs, crossn
    hless0 = np.where(h_cross < 0.0, 1, 0)
    heq0 = np.where(h_cross == 0.0, 1, 0)
    ne = np.where(yless0 != hless0, 1, 0)

    # print('{0:18} {1}'.format("yless0_cross shape: ", yless0.shape))

    disagree = np.sum(np.where(heq0 + ne > 0, 1.0, 0.0))

    # print("cnt: ", cnt, " disagree: ", disagree)
    print(float(cnt) / float(runs), disagree / (float(runs) * float(crossn)))
