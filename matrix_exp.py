# -*- encoding: utf-8 -*-

import numpy as np

def main():
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



if __name__ == "__main__":
    main()