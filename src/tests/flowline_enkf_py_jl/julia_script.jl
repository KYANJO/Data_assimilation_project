using ForwardDiff
using NLsolve
using Plots
using Statistics
using Random, Distributions, SignalAnalysis
using LinearAlgebra


#implicit flowline model function
function flowline!(F, varin, varin_old, params, grid, bedfun::Function)
    #Unpack grid
    NX = params["NX"]
    N1 = params["N1"]
    dt = params["dt"]/params["tscale"]
    ds = grid["dsigma"]
    sigma = grid["sigma"]
    sigma_elem = grid["sigma_elem"]

    #Unpack parameters
    tcurrent = params["tcurrent"]
    xscale = params["xscale"]
    hscale = params["hscale"]
    lambda = params["lambda"]
    m      = params["m"]
    n      = params["n"]
    a      = params["accum"]/params["ascale"]
    mdot   = params["facemelt"][tcurrent]/params["uscale"]
    eps    = params["eps"]
    transient = params["transient"]

    #Unpack variables
    h = varin[1:NX]
    u = varin[NX+1:2*NX]
    xg = varin[2*NX+1]

    h_old = varin_old[1:NX]
    xg_old = varin_old[2*NX+1]

    #Calculate bed
    hf = -bedfun(xg*xscale,params)/(hscale*(1-lambda))
    hfm = -bedfun(xg*last(sigma_elem)*xscale,params)/(hscale*(1-lambda))
    b =  -bedfun(xg.*sigma.*xscale,params)./hscale
    

    #Calculate thickness functions
    F[1]      = transient*(h[1]-h_old[1])/dt + (2*h[1]*u[1])/(ds[1]*xg) - a
    F[2]      = transient*(h[2]-h_old[2])/dt -
                    transient*sigma_elem[2]*(xg-xg_old)*(h[3]-h[1])/(2*dt.*ds[2]*xg) +
                        (h[2]*(u[2]+u[1]))/(2*xg*ds[2]) - a
    F[3:NX-1] = transient.*(h[3:NX-1] .- h_old[3:NX-1])./dt .-
                    transient.*sigma_elem[3:NX-1].*(xg - xg_old).*(h[4:NX].-h[2:NX-2])./(2 .* dt .* ds[3:NX-1] .* xg) .+
                        (h[3:NX-1] .* (u[3:NX-1] .+ u[2:NX-2]) .- h[2:NX-2] .* (u[2:NX-2] .+ u[1:NX-3]))./(2 .* xg .* ds[3:NX-1]) .- a
    F[N1] = (1+0.5*(1+(ds[N1]/ds[N1-1])))*h[N1] - 0.5*(1+(ds[N1]/ds[N1-1]))*h[N1-1] - h[N1+1]
    F[NX]     = transient*(h[NX]-h_old[NX])/dt -
                    transient*sigma[NX]*(xg-xg_old)*(h[NX]-h[NX-1])/(dt*ds[NX-1]*xg) +
                        (h[NX]*(u[NX] + mdot*hf/h[NX] + u[NX-1]) - h[NX-1]*(u[NX-1]+u[NX-2]))/(2*xg*ds[NX-1]) - a

    #Calculate velocity functions
    F[NX+1]      = (4*eps/(xg*ds[1])^((1/n)+1))*(h[2]*(u[2]-u[1])*abs(u[2]-u[1])^((1/n)-1) -
                  h[1]*(2*u[1])*abs(2*u[1])^((1/n)-1)) - u[1]*abs(u[1])^(m-1) -
                  0.5*(h[1]+h[2])*(h[2]-b[2]-h[1]+b[1])/(xg*ds[1])
    F[NX+2:2*NX-1] = (4 .* eps ./(xg .* ds[2:NX-1]).^((1/n)+1)) .* (h[3:NX] .* (u[3:NX] .- u[2:NX-1]) .* abs.(u[3:NX].-u[2:NX-1]).^((1/n)-1) .-
                  h[2:NX-1] .* (u[2:NX-1] .- u[1:NX-2]) .* abs.(u[2:NX-1] .- u[1:NX-2]).^((1/n)-1)) .-
                  u[2:NX-1] .* abs.(u[2:NX-1]).^(m-1) .- 0.5 .* (h[2:NX-1] .+ h[3:NX]) .* (h[3:NX] .- b[3:NX] .- h[2:NX-1] .+ b[2:NX-1])./(xg .* ds[2:NX-1])
    F[NX+N1] = (u[N1+1]-u[N1])/ds[N1] - (u[N1]-u[N1-1])/ds[N1-1]
    F[2*NX]     = (1/(xg*ds[NX-1])^(1/n))*(abs(u[NX]-u[NX-1])^((1/n)-1))*(u[NX]-u[NX-1]) - lambda*hf/(8*eps)

    #Calculate grounding line functions
    F[2*NX+1]        = 3*h[NX] - h[NX-1] - 2*hf
end

function Jac_calc(huxg_old, params, grid, bedfun::Function, flowlinefun::Function)
    #Use automatic differentiation to calculate Jacobian for nonlinear solver (more accurate and faster than finite difference!)
    f = varin -> (F = fill(zero(promote_type(eltype(varin), Float64)), 2*params["NX"]+1); flowlinefun(F, varin, huxg_old, params, grid, bedfun); return F)
    J = fill(0.0, 2*params["NX"]+1, 2*params["NX"]+1)
    Jf! = (J,varin) -> ForwardDiff.jacobian!(J, f, varin)
    return Jf!
end

function flowline_run(varin, params, grid, bedfun::Function, flowlinefun::Function)
    
    nt = params["NT"]
    huxg_old = varin
    huxg_all = zeros(size(huxg_old,1),nt)
    #huxg_all[:,1] = huxg_old

    for i in 1:nt
        if !params["assim"]
            params["tcurrent"] = i
        end
        Jf! = Jac_calc(huxg_old, params, grid, bedfun, flowlinefun)
        solve_result=nlsolve((F,varin) ->flowlinefun(F, varin, huxg_old, params, grid, bedfun), Jf!, huxg_old,iterations=100)
        huxg_old = solve_result.zero
        huxg_all[:,i] = huxg_old

        #if !solve_result.f_converged
            #err="Solver didn't converge at time step " * string(i)
            #print(err)
        #end
        if !params["assim"]
            print("Step " * string(i) * "\n")
        end
    end
    return huxg_all
end


function bed(x,params)
    
    b = params["sillamp"] .* (-2*acos.((1 - params["sillsmooth"]).*sin.(π*x/(2*params["xsill"])))./π .- 1)
    return b
   
end

# Define the main function to handle the "True" and "Wrong" simulations
# function solve_system(huxg_old, params, grid, bedfun, flowline_fun)
#     # Jac_calc and flowline are assumed to be defined elsewhere in your Julia script
#     Jf = Jac_calc(huxg_old, params, grid, bedfun, flowline_fun)

#     # Solve the nonlinear system using nlsolve
#     solve_result = nlsolve((F, varin) -> flowline_fun(F, varin, huxg_old, params, grid, bedfun), Jf, huxg_old, iterations=1000)
#     huxg_out0 = solve_result.zero  # Initial output

#     ### "True" simulation ###
#     params["NT"] = 150
#     params["TF"] = params["year"] * 150
#     params["dt"] = params["TF"] / params["NT"]
#     params["transient"] = 1

#     # Setting up facemelt distribution
#     params["facemelt"] = [LinRange(5, 85, params["NT"] + 1);] / params["year"]
#     fm_dist = Normal(0, 20.0)
#     fm_truth = params["facemelt"]  # You can add random noise to facemelt if needed
#     params["facemelt"] = fm_truth

#     # Run the "True" simulation
#     huxg_out1 = flowline_fun(huxg_out0, params, grid, bedfun)  # Replace flowline_run with the appropriate function

#     ### "Wrong" simulation ###
#     # Set up facemelt for the "wrong" simulation
#     fm_wrong = [LinRange(5, 45, params["NT"] + 1);] / params["year"]
#     params["facemelt"] = fm_wrong

#     # Adjust initial conditions for the wrong simulation
#     h_wrong = huxg_out0[1:params["NX"]] * 0.8
#     xg_wrong = 1330e3 / params["xscale"]
#     huxg_init = vcat(h_wrong, huxg_out0[params["NX"]+1:2*params["NX"]], xg_wrong)

#     # Run the "Wrong" simulation
#     huxg_out2 = flowline_fun(huxg_init, params, grid, bedfun)  # Replace flowline_run with the appropriate function

#     return huxg_out0, huxg_out1, huxg_out2  # Return "initial",  "True" and "Wrong" simulation outputs
# end


# function solve_system(huxg_old, params, grid, bedfun, flowline_fun)
#     # Jac_calc is assumed to be defined somewhere in your Julia script
#     Jf = Jac_calc(huxg_old, params, grid, bedfun, flowline_fun)
    
#     # Solve using nlsolve
#     solve_result = nlsolve((F, varin) -> flowline_fun(F, varin, huxg_old, params, grid, bedfun), Jf, huxg_old, iterations=1000)
    
#     # Return the result (the solution from nlsolve)
#     return solve_result.zero
# end
