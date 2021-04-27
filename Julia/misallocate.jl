  using Random
  
  
  function these(bign)
    Random.seed!(105) # seed random number generator for reproducible tests: # 4 allocations, 152 bytes

    # Declarations and initializations: 16 allocations, 40.539 Kib
    misses::Int64 = 0
    disagree::Int64 = 0

    x = rand(bign, 2) # 1 allocation, 1.766 KiB -> this line + next
    x .= x .* 2.0 .- 1.0
    x_intercept = hcat(ones(bign), x)  # 2 allocations

    y = zeros(bign)  # 1 allocation, 896 bytes

    misclassified_guesses = zeros(Int, bign) # 1 allocation, 896 bytes

    test = falses(bign) # 2 allocations, 128 bytes
    fx1 = zeros(bign) # 1 allocation, 896 bytes
    h = zeros(bign)  # 1 allocation, 896 bytes

    # cross-validation set
    # crossn = 10 * bign
    # x_cross = zeros(Float64, crossn, 2)  
    # y_cross = zeros(Int, crossn)  
    # h_cross = zeros(Float64, crossn)

    # algorithm
    # pa = rand(2) # 1 allocation, 96 bytes
    # pa .= pa .* 2 .- 1
    # pb = rand(2)
    # pb .= pb .* 2 .- 1
    # slope = (pb[2] - pa[2]) / (pb[1] - pa[1])
    # intercept = pb[2] - pb[1] * slope

    fx1[:] .= x[:,1] .* slope .+ intercept


  end