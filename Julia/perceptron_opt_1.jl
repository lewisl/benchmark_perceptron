#= 
    Simulate data for perceptron learning.
    Calculate the perceptron weights.
    Simulate a cross-validation data set and evaluate perceptron weights.
    before optimization:  0.120430 seconds (1.20 M allocations: 113.246 MiB, 9.08% gc time)
    after optimization: 0.057215 seconds (102.02 k allocations: 10.904 MiB)
=#
using Printf
using Random

# const runs = 1000  # for benchmark should be 1000

function simper(bign::Int64, runs=1000)
    Random.seed!(105) # seed random number generator for reproducible tests


    # Declarations and initialization to set scope to function level
    misses::Int64 = 0
    disagree::Int64 = 0

    x = rand(bign, 2) 
    x .= x .* 2.0 .- 1.0
    # x_intercept = hcat(ones(bign), x)

    y = zeros(bign)
    misclassified_guesses = zeros(Int, bign)

    h = zeros(bign)
    w = [0.0, 0.0, 0.0]

    # simulate a cross-validation set
    crossn = 10 * bign
    x_cross = rand(Float64, crossn, 2)  
    x_cross .= x_cross .* 2.0 .- 1.0
    y_cross = zeros(Int, crossn)  
    h_cross = zeros(Float64, crossn)
        

    for k = 1:runs
        # create a cutting line for "true" f(x) based on 2 arbitrary points
        pa = rand(2) # 1 allocation, 96 bytes
        pa .= pa .* 2 .- 1
        pb = rand(2)  # 1 allocation, 96 bytes
        pb .= pb .* 2 .- 1
        slope = (pb[2] - pa[2]) / (pb[1] - pa[1])
        intercept = pb[2] - pb[1] * slope
        # println(k, " ", slope, " ", intercept) 
        w .= 0.0  # initial pla hypothesis with weights equal 0

        @views for i = 1:bign
            fx1 = x[i,1] .* slope .+ intercept
            # label each sample of x with its "true" y label
            y[i] = (x[i,2] .> fx1) ? 1 : -1 
            # initial hypothesis with weights = 0
            h[i] = w[1] * 1.0 + w[2] * x[i,1] + w[3] * x[i,2]
        end


        # perform pla to determine g
        @views while true
            misclassified_guesses .= 0
            number_misclassified = 0
            for i = 1:bign
                if (y[i] < 0) != (h[i] < 0) || h[i] == 0
                    number_misclassified += 1
                    misclassified_guesses[number_misclassified] = i
                end
            end

            if  number_misclassified == 0  
                # println("****** converged")
                break
            else
                misses += 1
                if number_misclassified == 1
                    pick = misclassified_guesses[1]
                else
                    pick = misclassified_guesses[rand(1:number_misclassified)]
                end
                # println("pick: ", pick)
                # println("length of y: ", length(y))
                # println(y)
                w[1] += y[pick]
                w[2] += y[pick] * x[pick, 1]
                w[3] += y[pick] * x[pick, 2]

                h[pick] = w[1] * 1.0 + w[2] * x[pick, 1] + w[3] * x[pick, 2] 
            end
        end # while true

        # simulate cross-validation
        for i = 1:crossn
            fx1_cross = slope * x_cross[i, 1] + intercept
            y_cross[i] = x_cross[i, 2] >= fx1_cross ? 1 : -1  # "true" label for each x

            # calculate hypothesis for cross validation set
            h_cross[i] =  w[1] * 1.0 + w[2] * x_cross[i, 1] + w[3] * x_cross[i, 2] 

            # test gross error rate
            if (y_cross[i] < 0) != (h_cross[i] < 0) || h_cross[i] == 0
                disagree += 1
            end
        end # for i

    end # for k...

    # calculate summary stats for the entire run
    avg_iters = misses / runs
    avg_disagree_pct = disagree / (runs * crossn)
    # @printf "%0.3f  %0.3f\n" avg_iters avg_disagree_pct
    return (avg_iters, avg_disagree_pct)
end  # function simper

# run it a specific number of times
# if length(ARGS) < 1  
#     simper(100)
# else
#     try
#         xt::Int64 = ARGS[1]
#     catch
#         println()
#         println("First input argument must be an integer. Exiting.")
#         exit(1)
#     end
#     simper(xt)
# end # if length