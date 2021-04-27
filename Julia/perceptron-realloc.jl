#=
    Simulate data for perceptron learning.
    Calculate the perceptron weights.
    Simulate a cross-validation data set and evaluate perceptron weights.
=#

const runs = 1000  # for benchmark should be 1000
type point
        x1::Float64
        x2::Float64

        point() = new(rand() .* 2.0 .- 1.0, rand() .* 2.0 .- 1.0)
end

function reinitarray!(x)
    for j in eachindex(x)
       x[j] = 0.0
    end
end


function simper(bign::Int64=10)

    # Declarations and initialization to set scope to function level
    misses = 0
    disagree = 0
    crossn = 10 * bign
    srand(1) # seed random number generator for reproducible tests

    x = ones(Float64, bign, 3) # x[1] = intercept term = 1, x[2] = x1, x[3] = x2
    y = zeros(Int64, bign)
    misclassified = zeros(Int64, bign) # hold index of misclassified points

    for k = 1:runs
        # create a cutting line for "true" f(x) based on 2 arbitrary points
        pa = point()
        pb = point()
        slope = (pb.x2 - pa.x2) / (pb.x1 - pa.x2)
        intercept = pb.x2 - pb.x1 * slope
        # println(k, " ", slope, " ", intercept)

        # create the simulated dataset
        for i = 1:bign
            x[i,2] = rand() * 2.0 - 1.0
            x[i,3] = rand() * 2.0 - 1.0
            fx = slope * x[i, 2] + intercept
            # label each sample of x with its "true" y label
            y[i] = x[i, 3] >= fx ? 1 : -1
        end

        # initial pla hypothesis with weights equal 0
        w = [0.0, 0.0, 0.0]
        h = x * w   # initial pla hypothesis
        
        # perform pla to improve hypthesis
        while_limit = 0
        while true
            cnt_misclassified = 0
            fill!(misclassified, 0) # re-initialize array in place
            for i = 1:bign
                if (y[i] < 0) != (h[i] < 0) || h[i] == 0
                    cnt_misclassified += 1
                    misclassified[cnt_misclassified] = i
                end
            end
            # println("no. misclassified: ", cnt_misclassified)

            if cnt_misclassified == 0
                break
                # converged--but, also set limit at bottom of while to stop
            else
                misses += 1
                if cnt_misclassified == 1
                    pick = misclassified[1] # pick the only one
                else
                    pick = misclassified[rand(1:cnt_misclassified)] # use random int
                end

                # new hypothesis
                w[1] += y[pick]
                w[2] += y[pick] * x[pick, 2]
                w[3] += y[pick] * x[pick, 3]
                h[pick] = w[1] * 1.0 + w[2] * x[pick, 2] + w[3] * x[pick, 3]
            end # if cnt_misclassified

            if while_limit > 5 * bign
                println("**** Iteration $k failed to converge in $(5 * bign) iterations.")
                break
            end
            while_limit += 1
        end # while true

        # create cross-validation set and
        # calculate hypothesis for cross validation set

        for i = 1:crossn
            # simulate a cross-validation set
            # don't need entire matrix because we can process one point at a time
            # matrix has no perf. hit, just needs a little more memory

            xn2 = rand() * 2.0 - 1.0  # like column 2 of test set
            xn3 = rand() * 2.0 - 1.0  # like column 3 of test set
            fx = slope * xn2 + intercept
            yn = xn3 >= fx ? 1 : -1
            hn = w[1] + w[2] * xn2 + w[3] * xn3 # intercept term implicit xn1 = 1
            # accumulate disagreements
            if (yn < 0) != (hn < 0.0) || hn == 0.0
                disagree += 1
            end
        end # for i

    end # for k...

    # calculate summary stats for the entire run
    avg_iters = misses / runs
    avg_disagree_pct = disagree / (runs * crossn)
    @printf "%0.3f  %0.3f\n" avg_iters avg_disagree_pct
    return (avg_iters, avg_disagree_pct)

end  # function simper

# run it a specific number of times
if length(ARGS) < 1
    simper(100)
else
    try
        xt = parse(Int,ARGS[1])
        simper(xt)
    catch
        println("*** First input argument must be an integer. Exiting.")
        exit()
    end 
end # if length
