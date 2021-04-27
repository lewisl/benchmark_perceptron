

function test_scope(x::Int64)
    misses = 0
    disagreements = 0
    for i = 1:x
        while true
            misses += 1
            if i % 3 == 0
                disagreements += 1
            end
            if misses >= x
                break
            end
        end
    end
    println(misses, " ", disagreements, " ", disagreements / misses)
end    

test_scope(20)