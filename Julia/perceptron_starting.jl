#= 
    Simulate data for perceptron learning.
    Calculate the perceptron weights.
    Simulate a cross-validation data set and evaluate perceptron weights.
=#

const runs = 1000  # for benchmark should be 1000

function simper(bign::Int64)

    # Declarations and initialization to set scope to function level
    misses = 0
    disagree = 0
    crossn = 0
    srand(1) # seed random number generator for reproducible tests

    for k = 1:runs
        # create a cutting line for "true" f(x) based on 2 arbitrary points
        pa = rand(Float64,2) .* 2 .- 1
        pb = rand(Float64,2) .* 2 .- 1
        slope = (pb[2] - pa[2]) / (pb[1] - pa[1])
        intercept = pb[2] - pb[1] * slope
        # println(k, " ", slope, " ", intercept) 

        # create the simulated dataset

        x = zeros(Float64, bign, 2)
        y = zeros(Float64, bign, 1)

        # label each sample of x with its "true" y label
        for i = 1:bign
            x[i, :] = [(2 * rand() - 1.0) (2 * rand() - 1.0)]
            fx = slope * x[i, 1] + intercept
            y[i] = x[i, 2] >= fx ? 1 : -1
        end

        # initial pla hypothesis with weights equal 0
        w = [0.0, 0.0, 0.0]
        h = Array(Float64,bign)
        for i in 1:bign
            h[i] = (  w[1]*1.0 + w[2]*x[i, 1] + w[3]*x[i, 2] )
        end

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
                    pick = rand(1:number_misclassified) # choose a random integer
                    # println(misclassified)
                    pick = misclassified[pick]
                    # println("pick: ", pick)
                end
                # println("length of y: ", length(y))
                # println(y)
                w[1] += y[pick,1]
                w[2] += y[pick] * x[pick, 1]
                w[3] += y[pick] * x[pick, 2]

                h[pick] = w[1] * 1.0 + w[2] * x[pick, 1] + w[3] * x[pick, 2] 
            end
        end # while true

        # simulate a cross-validation set
        crossn = 10 * bign
        x = zeros(Float64, crossn, 2)  
        y = zeros(Float64, crossn)  

        for i = 1:crossn
            x[i, :] = [(2 * rand()-1.0) (2 * rand()-1.0)]
            fx = slope * x[i, 1] + intercept
            y[i] = x[i, 2] >= fx ? 1 : -1  # "true" label for each x
        end

        # calculate hypothesis for cross validation set
        h = zeros(Float64, crossn)
        for i = 1:crossn
            h[i] =  w[1] * 1.0 + w[2] * x[i, 1] + w[3] * x[i, 2] 
        end

        # test if same answer including this in previous loop:  should be same
        for i = 1:crossn
            if (y[i] < 0) != (h[i] < 0) || h[i] == 0
                disagree += 1
            end
        end
    end # for k...

    # calculate summary stats for the entire run
    avg_iters = misses / runs
    avg_disagree_pct::Float64 = disagree / (runs * crossn)
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