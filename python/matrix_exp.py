# -*- encoding: utf-8 -*-

import numpy as np

def einsum_tst():
    m1 = np.arange(15).reshape((5,3))
    m2 = 2 * m1
    v1 = np.array((1, 0.5, 0.5))
    v2 = np.array((2, .25, .25))
    res1 = m1.dot(v1)
    res2 = m2.dot(v2)

    mcombo = np.array((m1,m2))
    print(mcombo)
    vcombo = np.array((v1, v2))
    print()
    print(vcombo)
    rescombo = np.einsum("ijk, ik -> ij", mcombo, vcombo)
    print()
    print(res1)
    print()
    print(res2)
    print()
    print(rescombo)


runs = 1000
bign = 1000
cols = 3


def asone1_tst():  # fastest by a hair
    ma = np.ones((runs, bign, cols))
    ma[:, :, 1:] = np.random.uniform(-1.0, 1.0, (runs, bign, cols - 1))
    # print(ma)


def asone2_tst():  # slowest
    ma = np.random.uniform(-1.0, 1.0, (runs, bign, cols))
    ma[:, :, 0] = np.ones((runs, bign))
    # print(ma)


def astwo_tst():
    mb = np.random.uniform(-1.0, 1.0, (runs, bign, cols - 1))
    mb = np.concatenate((np.ones((runs, bign, 1)), mb), axis=2)
    # print(mb)
