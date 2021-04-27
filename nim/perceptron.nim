import os
import strutils  
import algorithm  # contains fill proc
import random

## Simulate data for perceptron learning.
## Calculate the perceptron weights.
## Simulate a cross-validation data set and evaluate perceptron weights.
proc simper(bign: int, runs: int = 1000): tuple[avg_iters, avg_disagree_pct : float ] =

    type
        fpoint = tuple[x1: float, x2: float]

    var
        misses:                 int = 0
        disagree:               int = 0
        crossn:                 int
        pick:                   int
        x:                      seq[fpoint]
        y:                      seq[int]
        h:                      seq[float]
        misclassified_guesses:  seq[int]
        number_misclassified:   int = 0
        w:                      array[0..2, float]
        pa:                     fpoint
        pb:                     fpoint
        fx1:                     float
        slope:                  float
        intercept:              float

        x_cross:                seq[fpoint]
        y_cross:                seq[int]
        h_cross:                seq[float]
        fx1_cross:              float


    # initialize variables
    newseq(x, bign)
    for i in 0 .. bign-1: x[i] = (rand(2.0)-1.0, rand(2.0)-1.0)

    newseq(y, bign)
    w = [0.0, 0.0, 0.0]
    newseq(h, bign)
    newseq(misclassified_guesses, bign)

    # initialize variables for simulated cross-validation set
    crossn = 10 * bign
    newseq(x_cross, crossn)  
    for i in 0 .. crossn-1: x_cross[i] = (rand(2.0)-1.0, rand(2.0)-1.0)   

    newseq(y_cross, crossn)  
    newseq(h_cross, crossn)  # element each time through loop???

    randomize(105)    

    for k in 1 .. runs:
        # create a cutting line for "true" f(x) based on 2 arbitrary points
        pa = (rand(2.0)-1.0, rand(2.0)-1.0)
        pb = (rand(2.0)-1.0, rand(2.0)-1.0)
        slope = (pb[1] - pa[1]) / (pb[0] - pa[0])
        intercept = pb[1] - pb[0] * slope
        fill(w, 0.0) # initialize weights each run for new cutting line

        for i in 0 .. bign-1:
            fx1 = x[i][0] * slope + intercept
            # label each sample of x with its "true" y label
            y[i] = if (x[i][1] >= fx1): 1  else: -1
            h[i] = w[0]*1.0 + w[1]*x[i][0] + w[2]*x[i][1] 

        # perform pla to determine g
        while true:
            fill(misclassified_guesses, 0)
            number_misclassified = 0

            for i in 0..bign-1:
                if (y[i] < 0) != (h[i] < 0) or h[i] == 0:
                    misclassified_guesses[number_misclassified] = i  
                    number_misclassified += 1

            if number_misclassified == 0:
                # echo "****** converged"
                break
            else:
                if number_misclassified == 1:
                    misses += 1
                    pick = misclassified_guesses[0]
                else:
                    misses += 1
                    pick = misclassified_guesses[rand(number_misclassified-1)]

                w[0] += float(y[pick])
                w[1] += float(y[pick]) * x[pick][0]
                w[2] += float(y[pick]) * x[pick][1]

                h[pick] = w[0] * 1.0 + w[1] * x[pick][0] + w[2] * x[pick][1] 

        for i in 0 .. crossn-1:
            fx1_cross = slope * x_cross[i][0] + intercept
            y_cross[i] = if x_cross[i][1] >= fx1_cross: 1  else: -1  # "true" label for each x

            # calculate hypothesis for cross validation set
            h_cross[i] =  w[0] * 1.0 + w[1] * x_cross[i][0] + w[2] * x_cross[i][1] 

            # test gross error rate
            if (y_cross[i] < 0) != (h_cross[i] < 0) or h_cross[i] == 0:
                disagree += 1

    # calculate summary stats for the entire run
    var avg_iters = tofloat(misses) / tofloat(runs)
    var avg_disagree_pct = tofloat(disagree) / (tofloat(runs) * tofloat(crossn))
    echo avg_iters, " ", avg_disagree_pct
    return (avg_iters, avg_disagree_pct)

# run it and ignore the return value
if paramCount() == 1:
    discard simper(parseInt(paramStr(1)))
elif paramCount() == 2:
   discard simper(parseInt(paramStr(1)), parseInt(paramStr(2)))
else:    
    quit("Requires exactly 1 or 2 integer parameters. Exiting.")