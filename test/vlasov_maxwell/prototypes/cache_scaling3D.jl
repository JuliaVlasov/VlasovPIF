include("pif3d.jl");




Np=Int(1e5) #Number of particles

using BenchmarkTools
#Initial poisson solve
# @benchmark rho=accum_osde2(pm,zn[:,1],zn[:,2],zn[:,3],zn[:,4])
#
# b=@belapsed rho=accum_osde($pm,$zn[:,1],$zn[:,2],$zn[:,3],$zn[:,4])
#
# time(b)




x1n=rand(Float64,Np)
x2n=rand(Float64,Np)
x3n=rand(Float64,Np)
x4n=rand(Float64,Np)

pm=pif3d{Float64}(1,1,1,1.0)
rho=Array(Float64,1,1,1)


mem=Array(Float64,0) #bytes
walltime=Array(Float64,0) #nano seconds (10^-9 s)
dofs=Array(Float64,0)
idx=2
for idx=0:11
  Nx1=2^(idx)
  Nx2=2
  Nx3=0
  #Particle mesh coupling
  pm=pif3d{Float64}(Nx1,Nx2,Nx3,1.0)
  rho=Array(Float64, pm.N1,pm.N2,pm.N3)

  result=@benchmark rho=accum_osde(pm,x1n,x2n,x3n,x4n)

  # @benchmark rho=accum_osde($pm,$zn[:,1],$zn[:,2],$zn[:,3],$zn[:,4])
  push!(walltime,minimum(result).time)
  push!(mem, minimum(result).memory)
  push!(dofs, pm.N1*pm.N2*pm.N3)

  print(result,"\n")
  print("dofs:", dofs[end],", memory:", mem[end],"\n")

end

pm=pif3d{Float64}(1,0,0,1.0)


using PyPlot
plot(mem/1e3,walltime/1e6,"-o")
plot(64*[1., 1.], [walltime[1] ,walltime[end]]/1e6)
xlabel("memory [kB]")
ylabel("wall time [ms]")
grid()

# loglog(dofs,walltime/1e6,"-o")
