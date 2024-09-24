

function f!(x,y)
    x[1] = 42
    y = 7 + y
    # return y
end

a = [4, 5, 6]
b = 3
# f(a, b)
# println(a,b)
f!(a,b)
println(a,b)

# form an array of 20 elements