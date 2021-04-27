#= 
    Simulate data for perceptron learning.
    Calculate the perceptron weights.
    Simulate a cross-validation data set and evaluate perceptron weights.
=#

const points = 1000  # for benchmark should be 1000

function simper(bign::Int64=10)

    # Declarations and initialization to set scope to function level
    misses = 0
    disagree = 0
    crossn = 0
    srand(1) # seed random number generator for reproducible tests

    for k = 1:points
        # create a cutting line for "true" f(x) based on 2 arbitrary points
        pa = rand(Float64,2) .* 2 .- 1
        pb = rand(Float64,2) .* 2 .- 1
        slope = (pb[2] - pa[2]) / (pb[1] - pa[1])
        intercept = pb[2] - pb[1] * slope
        # println(k, " ", slope, " ", intercept) 

        # create the simulated dataset

        x = rand(bign, 2) * 2.0 - 1.0 
        y = zeros(Int64, bign) 
        h = Array(Float64,bign) # pla hypothesis
        
        for i = 1:bign
            fx = slope * x[i, 1] + intercept
            # label each sample of x with its "true" y label
            y[i] = x[i, 2] >= fx ? 1 : -1
        end

        # initial pla hypothesis with weights equal 0
        w = [0.0, 0.0, 0.0]
        h = [ones(bign) x] * w

        # perform pla to determine g
        while true
            misclassified = Int64[]
            for i = 1:bign
                if (y[i] < 0) != (h[i] < 0) || h[i] == 0
                    push!(misclassified, i)  
                end
            end
            # println("no. misclassified: ", length(misclassified))
            number_misclassified = length(misclassified)
            if number_misclassified == 0
                # println("****** converged")
                break
            else
                misses += 1
                if number_misclassified == 1
                    pick = misclassified[1]
                else
                    pick::Int64 = rand(1:number_misclassified) # choose a random integer
                    # println(misclassified)
                    pick = misclassified[pick]
                    # println("y: ", y[pick])
                end
                # println("length of y: ", length(y))
                # println(y)
                w[1] += y[pick]
                w[2] += y[pick] * x[pick, 1]
                w[3] += y[pick] * x[pick, 2]

                h[pick] = w[1] * 1.0 + w[2] * x[pick, 1] + w[3] * x[pick, 2] 
            end
        end # while true

        # cross-validation set
        crossn = 10 * bign
        x = rand(crossn, 2) * 2.0 - 1.0
        y1 = 0
        # calculate hypothesis for cross validation set
        h = [ones(crossn) x] * w
        
        for i = 1:crossn
            # simulate a cross-validation set
            fx = slope * x[i, 1] + intercept
            y1 = x[i, 2] >= fx ? 1 : -1
            # accumulate disagreements
            if (y1 < 0) != (h[i] < 0) || h[i] == 0
                disagree += 1
            end
        end # for i

    end # for k...

    # calculate summary stats for the entire run
    avg_iters = misses / points
    avg_disagree_pct = disagree / (points * crossn)
    @printf "%0.3f  %0.3f\n" avg_iters avg_disagree_pct
    return (avg_iters, avg_disagree_pct)
end  # function simper

# run it a specific number of times
if length(ARGS) < 1  
    simper(100)
else
    try
        xt::Int64 = ARGS[1]
    catch
        println()
        println("First input argument must be an integer. Exiting.")
        exit(1)
    end
    simper(xt)
end # if length