import random
import os
import strutils  # use with fmtFloat to format individual value--using strfmt instead
import strformat  # to use Python style format codes

proc simper(bign: int): tuple[avg_iters, avg_disagree_pct : float ] =
    ## Simulate data for perceptron learning.
    ## Calculate the perceptron weights.
    ## Simulate a cross-validation data set and evaluate perceptron weights.

    type
        fpoint = tuple[x1: float, x2: float]

    var
        cnt:            int = 0
        disagree:       int = 0
        crossn:         int
        pick:           int
        x:              seq[fpoint]
        y:              seq[float]
        h:              seq[float]
        misclassified:  seq[int]
        w:              array[0..2, float]
        pa:             fpoint
        pb:             fpoint
        fx:             float
        slope:          float
        intercept:      float

    const
        runs = 1000

    randomize()    
    for k in 1 .. runs:
        # create a cutting line for "true" f(x) based on 2 arbitrary points
        pa = (rand(2.0)-1.0, rand(2.0)-1.0)
        pb = (rand(2.0)-1.0, rand(2.0)-1.0)
        slope = (pb[1] - pa[1]) / (pb[0] - pa[0])
        intercept = pb[1] - pb[0] * slope
        # echo k, " ", slope, " ", intercept

        # create the simulated dataset
        newseq(x, bign)
        newseq(y, bign)

        # label each sample of x with its "true" y label
        for i in 0 .. bign-1:
            x[i] = (rand(2.0)-1.0, rand(2.0)-1.0)
            fx = slope * x[i][0] + intercept
            y[i] = if x[i][1] >= fx: 1  else: -1
            # echo x[i], " ", fx, " ", y[i]

        # initial pla hypothesis with weights equal 0
        w = [0.0, 0.0, 0.0]
        newseq(h, bign)
        for i in 0 .. bign-1:
            h[i] = (  w[0]*1.0 + w[1]*x[i][0] + w[2]*x[i][1] )
            # h.add(  sum(w[0]*1.0, w[1]*x[i][0], w[2]*x[i][1])  )

        # perform pla to determine g
        while true:
            misclassified = @[]
            for i in 0 .. bign-1:
                if (y[i] < 0) != (h[i] < 0) or h[i] == 0:
                    misclassified.add(i)  # add means append to sequence

            if len(misclassified) == 0:
                # echo "****** converged"
                break
            else:
                cnt += 1
                if len(misclassified) == 1:
                    pick = misclassified[0]
                else:
                    pick = rand(len(misclassified)-1)
                    pick = misclassified[pick]

                w[0] += y[pick]
                w[1] += y[pick] * x[pick][0]
                w[2] += y[pick] * x[pick][1]

                h[pick] = w[0] * 1.0 + w[1] * x[pick][0] + w[2] * x[pick][1] 
                # h[pick] = sum( w[0]*1, w[1]*x[pick][0], w[2]*x[pick][1] ) did NOT work

        # simulate a cross-validation set
        crossn = 10 * bign
        newseq(x, crossn)  # prob faster than setting to empty (@[]) and adding 1 
        newseq(y, crossn)  # element each time through loop???

        randomize()
        for i in 0 .. crossn-1:
            x[i] = (rand(2.0)-1.0, rand(2.0)-1.0)
            fx = slope * x[i][0] + intercept
            y[i] = if x[i][1] >= fx: 1  else: -1  # "true" label for each x

        # calculate hypothesis for cross validation set
        newseq(h, crossn)
        for i in 0 .. crossn-1:
            h[i] =  w[0] * 1.0 + w[1] * x[i][0] + w[2] * x[i][1] 

        for i in 0 .. crossn-1:
            if (y[i] < 0) != (h[i] < 0) or h[i] == 0:
                disagree += 1

    # calculate summary stats for the entire run
    var avg_iters = tofloat(cnt) / tofloat(runs)
    var avg_disagree_pct = tofloat(disagree) / (tofloat(runs) * tofloat(crossn))
    echo (avg_iters, avg_disagree_pct)
    return (avg_iters, avg_disagree_pct)

# run it and ignore the return value
var xt: int
if paramCount() < 1:
    discard simper(10)
else:
    try:
        xt = parseInt(paramStr(1))
    except:
        # let
        #     e = getCurrentException()
        #     msg = getCurrentExceptionMsg()
        # echo "Got exception ", repr(e), " with message ", msg
        quit("Input argument must be an integer. Exiting.")
    discard simper(xt)