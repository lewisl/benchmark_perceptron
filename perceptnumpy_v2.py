#! /usr/bin/env python3.4
# -*- encoding: utf-8 -*-

import random as r
import numpy as np


def randper(bign):
    cnt = 0
    disagree = 0
    runs = 1000
    crossn = 10 * bign
    np.random.seed(1)
    r.seed(1)

    # fa = (r.uniform(-1, 1), r.uniform(-1, 1))   # runs x 2
    fa = np.random.uniform(-1, 1, (runs, 2))
    # fb = (r.uniform(-1, 1), r.uniform(-1, 1))
    fb = np.random.uniform(-1, 1, (runs, 2))
    slope = (fb[:, 1] - fa[:, 1]) / (fb[:, 0] - fa[:, 0])   # runs
    intercept = fb[:, 1] - fb[:, 0] * slope

    print("fa shape: ", fa.shape)
    print("fb shape: ", fb.shape)
    print("slope shape: ", slope.shape)
    print("intercept shape: ", intercept.shape)

    # create the simulated dataset
    x = 2.0 * np.random.rand(runs, bign, 2) - 1.0     # runs,bign,2
    fx = (np.repeat(slope[:,np.newaxis],bign,1) * x[:, :, 0]
          + np.repeat(intercept[:,np.newaxis], bign, 1))          # runs, bign
    y = np.where(x[:, :, 1] >= fx, 1.0, -1.0)    # runs, bign

    print("x shape: ", x.shape)
    print("fx shape: ", fx.shape)
    print("y shape: ", y.shape)

    # add column of ones
    x = np.concatenate((x, np.ones((runs, bign, 1))), axis=2)  # runs, bign, 3
    print("new x shape: ", x.shape)

    # Calculate PLA hypothesis values
    # initial pla hypothesis with weights equal 0
    w = np.array([0.0, 0.0, 0.0])
    h = x[:,].dot(w)                        # runs, bign
    print("h shape: ", h.shape)
    #
    # # perform pla to determine g
    # while True:
    #     misclassified = np.argwhere((np.where(y < 0.0, 1, -1)   # runs by variable
    #         * np.where(h < 0.0, 1, -1) + np.where(h == 0.0, -1, 0)) <= 0)
    #     if len(misclassified) == 0:
    #         break
    #     else:
    #         cnt += 1
    #         pick = r.randint(0, len(misclassified) - 1)
    #         pick = misclassified[pick][0]
    #
    #         # calc new weights and new hypothesis value
    #         w[0] += y[pick]
    #         w[1] += y[pick] * x[pick][1]
    #         w[2] += y[pick] * x[pick][2]
    #         h[pick] = x[pick].dot(w)
    #
    # # simulate a cross-validation set
    # # evaluate g on a different set of points than those used to estimate g
    # x_cross = 2.0 * np.random.rand(crossn, 2) - 1.0     # runs, crossn, 2
    # fx = slope * x_cross[:, 0] + intercept              # 1 x runs
    # y_cross = np.where(x_cross[:, 1] >= fx, 1, -1)      # runs, crossn, 1
    # # next, add the ones
    # x_cross = np.column_stack((np.ones((crossn, 1)), x_cross))  # runs, crossn, 3
    # h_cross = x_cross.dot(w)                                    # runs, crossn, 1
    #
    # yless0 = np.where(y_cross < 0.0, 1, 0)      # runs, crossn
    # hless0 = np.where(h_cross < 0.0, 1, 0)
    # heq0 = np.where(h_cross == 0.0, 1, 0)
    # ne = np.where(yless0 != hless0, 1, 0)
    # disagree += np.sum(np.where(heq0 + ne > 0, 1, 0))
    #
    # print(float(cnt) / float(runs), disagree / (float(runs) * float(crossn)))
