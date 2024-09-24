using ForwardDiff
using NLsolve
using Plots
using Statistics
using Random, Distributions, SignalAnalysis
using LinearAlgebra

#Prescribed initial values of model parameters in a dictionary
function params_define()    
    params = Dict()
    params["A"] = 4e-26
    params["year"] = 3600 * 24 * 365
    params["n"] = 3
    params["C"] = 3e6
    params["rho_i"] = 900
    params["rho_w"] = 1000
    params["g"] = 9.8
    params["B"] = params["A"]^(-1 / params["n"])
    params["m"] = 1 / params["n"]
    params["accum"] = 0.65 / params["year"]
    params["facemelt"] = 5 / params["year"]

    #Scaling parameters
    params["hscale"] = 1000
    params["ascale"] = 1.0 / params["year"]
    params["uscale"] =(params["rho_i"] * params["g"] * params["hscale"] * params["ascale"] / params["C"])^(1 / (params["m"] + 1))
    params["xscale"] = params["uscale"] * params["hscale"] / params["ascale"]
    params["tscale"] = params["xscale"] / params["uscale"]
    params["eps"] = params["B"] * ((params["uscale"] / params["xscale"])^(1 / params["n"])) / (2 * params["rho_i"] * params["g"] * params["hscale"])
    params["lambda"] = 1 - (params["rho_i"] / params["rho_w"])

    #Grid parameters
    params["NT"] = 1
    params["TF"] = params["year"]
    params["dt"] = params["TF"] / params["NT"]
    params["transient"] = 0
    params["tcurrent"] = 1

    params["N1"] = 40
    params["N2"] = 10
    params["sigGZ"] = 0.97
    params["NX"] = params["N1"] + params["N2"]

    #Bed params
    #params["b0"] = -400
    params["xsill"] = 50e3
    params["sillamp"] = 500
    params["sillsmooth"] = 1e-5
    #params["bxr"] = 1e-3
    #params["bxp"] = -1e-3

    #EnKF params
    params["inflation"] = 1.0
    params["assim"] = false

    #sigma = vcat(LinRange(0, 1, convert(Integer, params["N1"])));
    sigma1=LinRange(params["sigGZ"]/(params["N1"]+0.5), params["sigGZ"], convert(Integer, params["N1"]))
    sigma2=LinRange(params["sigGZ"], 1, convert(Integer, params["N2"]+1))
    sigma = vcat(sigma1,sigma2[2:params["N2"]+1])
    grid = Dict("sigma" => sigma)
    grid["sigma_elem"] = vcat(0,(sigma[1:params["NX"]-1] + sigma[2:params["NX"]]) ./ 2)
    grid["dsigma"] = diff(grid["sigma"])
    return params, grid
end

#Define bed topography function
function bed(x,params)
    #b = zeros(size(x))
    #if size(x,1)==1 & any(x.<params["xsill"])
    #    b = params["b0"] + params["bxr"].*x
    #elseif size(x,1)==1 & any(x.≥params["xsill"])
    #    b = params["b0"] + params["bxr"].*params["xsill"] + params["bxp"].*(x.-params["xsill"])
    #else
    #    b[x.<params["xsill"]] = params["b0"] .+ params["bxr"].*x[x.<params["xsill"]]
    #    b[x.≥params["xsill"]] = params["b0"] .+ params["bxr"].*params["xsill"] .+ params["bxp"].*(x[x.≥params["xsill"]].-params["xsill"])
    #end
    b = params["sillamp"] .* (-2*acos.((1 - params["sillsmooth"]).*sin.(π*x/(2*params["xsill"])))./π .- 1)
    return b
    #729 .- 2184.8.*(x./750e3).^2 .+ 1031.72.*(x/750e3).^4 .- 151.72.*(x./750e3).^6
end

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

#Analysis step of EnKF
function EnKF(huxg_ens,huxg_obs,ObsFun::Function,JObsFun::Function,Cov_obs,Cov_model,params)
    n = size(huxg_ens)[1] #state size
    N = size(huxg_ens)[2] #ensemble size 
    m = size(huxg_obs)[1] #measurement size

    huxg_ens_mean = mean(huxg_ens,dims=2) #mean of model forecast ensemble
    Jobs = JObsFun(n,params["m_obs"]) #Jacobian of observation operator
    
    KalGain = Cov_model*transpose(Jobs)*inv(Jobs*Cov_model*transpose(Jobs) + Cov_obs) #compute Kalman gain

    obs_virtual = zeros(m,N)
    analysis_ens = zeros(n,N)
    #mismatch_ens = zeros(n-1,N)
    for i in 1:N
        obs_virtual[:,i]  = huxg_obs + rand(MvNormal(zeros(m), Cov_obs))  #generate virtual observations
        analysis_ens[:,i] = huxg_ens[:,i] + KalGain * (obs_virtual[:,i]-ObsFun(huxg_ens[:,i],params["m_obs"])) #generate analysis ensemble
        #mismatch_ens[:,i] = obs_virtual[:,i]-ObsFun(huxg_ens[:,i],params["m_obs"]) #calculate mismatch
    end

    analysis_ens_mean = mean(analysis_ens,dims=2) #mean of analysis ensemble
    
    analysis_cov = (1/(N-1)) * (analysis_ens .- repeat(analysis_ens_mean,1,N)) * transpose(analysis_ens .- repeat(analysis_ens_mean,1,N)) #analysis error covariance
    analysis_cov = analysis_cov.*taper
    return analysis_ens, analysis_cov
end

function Obs(huxg_virtual_obs,m_obs)
    #huxg_obs = huxg_model

    n = size(huxg_virtual_obs)[1]
    m = m_obs
    H = zeros(m*2+1,n)
    di = convert(Int16, (n-2)/(2*m))  #distance between measurements
    for i in 1:m
        H[i,i*di] = 1
        H[m+i,convert(Int16, ((n-2)/2)+(i*di))] = 1
    end
    H[m*2+1,n-1] = 1
    z = H*huxg_virtual_obs
    return z
end

function JObs(n_model,m_obs)
    #Matrix(1.0I, size(huxg_model,1), size(huxg_model,1))

    n = n_model
    m = m_obs
    H = zeros(m*2+1,n)
    di = convert(Int16, (n-2)/(2*m))  #distance between measurements
    for i in 1:m
        H[i,i*di] = 1
        H[m+i,convert(Int16, ((n-2)/2)+(i*di))] = 1
    end
    H[m*2+1,n-1] = 1
    return H
end

#Initial guess and steady-state
params, grid = params_define()
xg = 300e3/params["xscale"]
hf = (-bed(xg*params["xscale"],params)/params["hscale"])/(1-params["lambda"])
h  = 1 .- (1-hf).*grid["sigma"]
u  = 1.0.*(grid["sigma_elem"].^(1/3)) .+ 1e-3
huxg_old = vcat(h,u,xg)

Jf! = Jac_calc(huxg_old, params, grid, bed, flowline!)
solve_result=nlsolve((F,varin) ->flowline!(F, varin, huxg_old, params, grid, bed), Jf!, huxg_old,iterations=1000)
huxg_out0 = solve_result.zero
#huxg_out0 = flowline_run(huxg_old, params, grid, bed, flowline!)

# Plot stuff
#x = [LinRange(0,500e3,convert(Integer, 1e3));]
#h0 = huxg_out0[1:params["NX"]].*params["hscale"];
#xg0 = huxg_out0[2*params["NX"]+1].*params["xscale"];
#hl0=vcat(h0.+bed(grid["sigma_elem"].*xg0,params),bed(xg0,params))
#xl0=vcat(grid["sigma_elem"],1).*xg0./1e3

#h1 = huxg_out1[1:params["NX"]].*params["hscale"];
#xg1 = huxg_out1[2*params["NX"]+1].*params["xscale"];
#hl1=vcat(h1.+bed(grid["sigma_elem"].*xg1,params),bed(xg1,params))
#xl1=vcat(grid["sigma_elem"],1).*xg1./1e3
#plot(x./1e3,bed(x,params),lw=3,linecolor=:tan4)
#plot!(xl0,hl0,lw=3,linecolor=:blue)
#plot!(xl1,hl1,lw=3,linecolor=:red)

#"True" simulation
#params["accum"] = 0.65 / params["year"]
params["NT"] = 150
params["TF"] = params["year"]*150
params["dt"] = params["TF"] / params["NT"]
params["transient"] = 1
params["facemelt"] = [LinRange(5,85,params["NT"]+1);]  / params["year"]
fm_dist = Normal(0,20.0)
#params["facemelt"] = params["facemelt"] + (rand(fm_dist, size(params["facemelt"])) / params["year"])
fm_truth = params["facemelt"] #+ (rand(RedGaussian(size(params["facemelt"],1),20.0)) / params["year"])
params["facemelt"] = fm_truth
huxg_out1 = flowline_run(huxg_out0, params, grid, bed, flowline!)

#"Wrong" simulation
#h_wrong = huxg_out0[1:params["NX"]] .* 0.8;
#xg_wrong = 1330e3/params["xscale"];
#huxg_init = vcat(h_wrong,huxg_out0[params["NX"]+1:2*params["NX"]],xg_wrong)
fm_wrong =[LinRange(5,45,params["NT"]+1);]  / params["year"]
params["facemelt"] = [LinRange(5,45,params["NT"]+1);]  / params["year"]
huxg_out2 = flowline_run(huxg_out0, params, grid, bed, flowline!)

ts = LinRange(0,params["TF"]/params["year"],params["NT"]+1)
xg_truth = vcat(huxg_out0[2*params["NX"]+1],huxg_out1[2*params["NX"]+1,:]).*params["xscale"];
xg_wrong = vcat(huxg_out0[2*params["NX"]+1],huxg_out2[2*params["NX"]+1,:]).*params["xscale"];
plot(ts,xg_truth./1e3,lw=3,linecolor=:black,lab="truth")
plot!(ts,xg_wrong./1e3,lw=3,linecolor=:red,lab="wrong")
plot!(ts,250.0.*ones(size(ts)),lw=1,linecolor=:black,linestyle=:dash,lab="sill")

#Set ensemble parameters
statevec_init = vcat(huxg_out0,params["facemelt"][1]/params["uscale"])
nd = size(statevec_init,1)  #dimension of model state
N = 30                  #number of ensemble members

sig_model = 1e-1
sig_obs = 1e-2
sig_Q = 1e-2
global Cov_model = (sig_model^2)*Matrix(1.0I, nd, nd) #placeholder until first filter update
Q = (sig_Q^2)*Matrix(1.0I, nd, nd)
#Q[end,end] = 20*(sig_Q^2)

#Q[nd-1,nd-1] = (sig_Q^2)

#Set model parameters for single time step runs
nt = params["NT"]
tfinal_sim = params["TF"]
ts = [(0.0:1.0:params["NT"]);].*params["year"]
params["NT"] = 1
params["TF"] = params["year"]*1
params["dt"] = params["TF"] / params["NT"]
params["transient"] = 1
params["assim"] = true

statevec_sig = vcat(grid["sigma_elem"],grid["sigma"],1,1)
#taper = zeros(size(statevec_sig,1),size(statevec_sig,1))
#for i in 1:size(taper,1)
#    for j in 1:size(taper,2)
#        sigdist = abs(statevec_sig[i] - statevec_sig[j])
#        if sigdist<0.2
#            taper[i,j] = exp(-500*(sigdist)^2)
#        end
#   end
#end
taper = ones(size(statevec_sig,1),size(statevec_sig,1))
#taper[1:end-1,end] = zeros(size(statevec_sig,1)-1,1)
#taper[end,1:end-1] = zeros(1,size(statevec_sig,1)-1)
taper[end,end-2] = 2
taper[end-2,end] = 2
taper[end,end] = 10
taper[end-1,end] = 10
taper[end,end-1] = 10

#Generate synthetic observations of thickness from "truth" simulation
ts_obs = [(10.0:10.0:140.0);].*params["year"]
idx_obs = findall(in(ts_obs),ts)
obs_dist = Normal(0,sig_obs)
huxg_virtual_obs = vcat(huxg_out1[:,idx_obs],Transpose(fm_truth[idx_obs]/params["uscale"]))
huxg_virtual_obs = huxg_virtual_obs + rand(obs_dist, size(huxg_virtual_obs))
#huxg_obs[end,:] = Obs(huxg_out1[end,idx_obs]) + 2*rand(obs_dist, size(huxg_out1[end,idx_obs]))
params["m_obs"] = 10  

#Initialize ensemble
statevec_bg = zeros(nd,nt+1) #ub
statevec_ens_mean = zeros(nd,nt+1) #ua
mm_ens_mean = zeros(nd-1,nt+1) #ua
statevec_ens = zeros(nd,N)  #uai
statevec_ens_full = zeros(nd,N,nt+1) #uae

statevec_bg[:,1] = statevec_init
statevec_ens_mean[:,1] = statevec_init
for i in 1:N
    statevec_ens[1:end-1,i] = statevec_init[1:end-1] + rand(MvNormal(zeros(nd-1), Cov_model[1:end-1,1:end-1]))
    statevec_ens[end,i] = statevec_init[end]
end

statevec_ens_full[:,:,1] = statevec_ens

#Run ensemble with assimilation (note obs available at only some times)
for k in 1:nt
    params["tcurrent"] = k
    print("Step " * string(k) * "\n")
    #forecast
    statevec_bg[1:end-1,k+1] = flowline_run(statevec_bg[1:end-1,k], params, grid, bed, flowline!) #background trajectory [without correction]
    statevec_bg[end,k+1] = params["facemelt"][k+1]/params["uscale"]

    for i in 1:N  #forecast ensemble
        huxg_temp = flowline_run(statevec_ens[1:end-1,i], params, grid, bed, flowline!)
        global nos = rand(MvNormal(zeros(nd), Q))
        #global nos[params["NX"]+1:2*params["NX"]+1] = zeros(params["NX"]+1,1)
        #global nos[1:params["NX"]] = rand(MvNormal(zeros(1), sig_Q^2)).*ones(params["NX"],1)
        global statevec_ens[:,i] = vcat(huxg_temp,params["facemelt"][k+1]/params["uscale"]) + nos
    end

    global statevec_ens_mean[:,k+1] = mean(statevec_ens,dims=2) # compute the mean of analysis ensemble
    if ~isempty(findall( x -> x == ts[k+1], ts_obs))
        global idx_obs = findall( x -> x == ts[k+1], ts_obs)
        global Cov_model = (1/(N-1)) * (statevec_ens .- repeat(statevec_ens_mean[:,k+1], 1, N)) * transpose(statevec_ens .- repeat(statevec_ens_mean[:,k+1], 1, N)) #forecast error covariance matrix
        global Cov_model = Cov_model.*taper
        global Cov_obs = (sig_obs^2)*Matrix(1.0I, 2*params["m_obs"]+1, 2*params["m_obs"]+1) #measurement noise covariance
        global huxg_obs = Obs(huxg_virtual_obs[:,idx_obs],params["m_obs"])   #subsample virtual obs to actual measurement locations

        statevec_ens_temp,Cov_model = EnKF(statevec_ens,huxg_obs,Obs,JObs,Cov_obs,Cov_model,params) #analysis corrections
        global statevec_ens = statevec_ens_temp
        global statevec_ens_mean[:,k+1] = mean(statevec_ens,dims=2) # compute the mean of analysis ensemble
        #global mm_ens_mean[:,k+1] = mean(statevec_ens_mm,dims=2) # compute the mean of analysis ensemble
        global statevec_ens = repeat(statevec_ens_mean[:,k+1], 1, N) + params["inflation"]*(statevec_ens - repeat(statevec_ens_mean[:,k+1], 1, N))

        params["facemelt"][k+1:end] =  statevec_ens_mean[end,k+1]*params["uscale"].*ones(size(params["facemelt"][k+1:end])) #update param value (assume it holds at all future time steps)
    end
    
    global statevec_ens_full[:,:,k+1] = statevec_ens
end

# Plot GL stuff
huxg_obs = Obs(huxg_virtual_obs,params["m_obs"])

plot(layout=(4,1),size=(600,1000))

xg_idx = 2*params["NX"]+1
xg_truth = vcat(huxg_out0[xg_idx],huxg_out1[xg_idx,:]).*params["xscale"];
xg_wrong = vcat(statevec_init[xg_idx],huxg_out2[xg_idx,:]).*params["xscale"]
xg_EnKF_ens_mean = statevec_ens_mean[xg_idx,:].*params["xscale"]
xg_EnKF_ens = transpose(statevec_ens_full[xg_idx,:,:].*params["xscale"])
xg_obs = huxg_obs[end,:].*params["xscale"]

plot!(subplot=1,ts./params["year"],xg_EnKF_ens./1e3,lw=0.5,linecolor=:gray,lab=hcat("EnKF ens",fill("",1,N)))
plot!(subplot=1,ts./params["year"],xg_truth./1e3,lw=3,linecolor=:black,lab="truth")
plot!(subplot=1,ts./params["year"],xg_wrong./1e3,lw=3,linecolor=:red,lab="wrong")
plot!(subplot=1,ts./params["year"],xg_EnKF_ens_mean./1e3,lw=3,linecolor=:blue,lab="EnKF mean")
plot!(subplot=1,ts_obs./params["year"],xg_obs./1e3, seriestype=:scatter,lab="Obs")
xlabel!(subplot=1,"time (kyr)")
ylabel!(subplot=1,"GL position (km)")
plot!(subplot=1,legend=:outertop, legendcolumns=5)

# Plot h mid-profile
h_truth = vcat(huxg_out0[25],huxg_out1[25,:]).*params["hscale"];
h_wrong = vcat(statevec_init[25],huxg_out2[25,:]).*params["hscale"]
h_EnKF_ens_mean = statevec_ens_mean[25,:].*params["hscale"]
h_EnKF_ens = transpose(statevec_ens_full[25,:,:].*params["hscale"])
h_obs = huxg_virtual_obs[25,:].*params["hscale"]

plot!(subplot=2,ts./params["year"],h_EnKF_ens,lw=0.5,linecolor=:gray,lab=hcat("EnKF ens",fill("",1,N)))
plot!(subplot=2,ts./params["year"],h_truth,lw=3,linecolor=:black,lab="truth")
plot!(subplot=2,ts./params["year"],h_wrong,lw=3,linecolor=:red,lab="wrong")
plot!(subplot=2,ts./params["year"],h_EnKF_ens_mean,lw=3,linecolor=:blue,lab="EnKF mean")
plot!(subplot=2,ts_obs./params["year"],h_obs, seriestype=:scatter,lab="Obs")
plot!(subplot=2,legend=:outertop, legendcolumns=4)
xlabel!(subplot=2,"time (kyr)")
ylabel!(subplot=2,"h (m)")

# Plot u at terminus
u_truth = vcat(huxg_out0[100],huxg_out1[100,:]).*params["uscale"].*params["year"]
u_wrong = vcat(statevec_init[100],huxg_out2[100,:]).*params["uscale"].*params["year"]
u_EnKF_ens_mean = statevec_ens_mean[100,:].*params["uscale"].*params["year"]
u_EnKF_ens = transpose(statevec_ens_full[100,:,:].*params["uscale"]).*params["year"]
u_obs = huxg_virtual_obs[100,:].*params["uscale"].*params["year"]

plot!(subplot=3,ts./params["year"],u_EnKF_ens,lw=0.5,linecolor=:gray,lab=hcat("EnKF ens",fill("",1,N)))
plot!(subplot=3,ts./params["year"],u_truth,lw=3,linecolor=:black,lab="truth")
plot!(subplot=3,ts./params["year"],u_wrong,lw=3,linecolor=:red,lab="wrong")
plot!(subplot=3,ts./params["year"],u_EnKF_ens_mean,lw=3,linecolor=:blue,lab="EnKF mean",ylims=(0,1000))
plot!(subplot=3,ts_obs./params["year"],u_obs, seriestype=:scatter,lab="Obs")
plot!(subplot=3,legend=:outertop, legendcolumns=4)
xlabel!(subplot=3,"time (kyr)")
ylabel!(subplot=3,"u (m/yr)")

# Plot terminus melt
fm_EnKF_ens_mean = statevec_ens_mean[end,:].*params["uscale"].*params["year"]
fm_EnKF_ens = transpose(statevec_ens_full[end,:,:].*params["uscale"].*params["year"])

plot!(subplot=4,ts./params["year"],fm_EnKF_ens,lw=0.5,linecolor=:gray,lab=hcat("EnKF ens",fill("",1,N)))
plot!(subplot=4,ts./params["year"],fm_truth.*params["year"],lw=3,linecolor=:black,lab="truth")
plot!(subplot=4,ts./params["year"],fm_wrong.*params["year"],lw=3,linecolor=:red,lab="wrong")
plot!(subplot=4,ts./params["year"],fm_EnKF_ens_mean,lw=3,linecolor=:blue,lab="EnKF mean",ylims=(0,90))
plot!(subplot=4,legend=:outertop, legendcolumns=4)
xlabel!(subplot=4,"time (kyr)")
ylabel!(subplot=4,"Terminus Melt Rate (m/yr)")