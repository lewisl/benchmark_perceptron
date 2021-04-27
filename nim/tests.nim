import random

type
    fpoint = tuple[x1: float, x2: float]

var
    bign:                       int = 100
    # cnt:                      int = 0
    # disagree:                 int = 0
    # crossn:                   int
    # pick:                     int
    x:                          seq[fpoint]
    # y:                          seq[float]
    # h:                        seq[float]
    misclassified_guesses:      seq[int]
    # w:                        array[0..2, float]
    pa:             fpoint
    # pb:             fpoint
    # fx:             float
    # slope:          float
    # intercept:      float

randomize(45)
pa = (rand(2.0) - 1.0, rand(2.0) - 1.0)
echo pa

newseq(misclassified_guesses, bign)
for i in 0..<bign: 
    misclassified_guesses[i] = 0

echo misclassified_guesses


newseq(x, bign)
for i in 0 .. bign-1:
    x[i] = (rand(2.0)-1.0, rand(2.0)-1.0)

echo x

for i in 1 .. 20:
    echo rand(5-1)