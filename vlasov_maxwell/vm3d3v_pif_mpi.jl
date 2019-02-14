# 6D Vlasov - Poisson Single Species
# For mpi use
using MPI

include("pif_tools.jl")
include("particle_sampling.jl")
importall pif

using psmp
using ProgressMeter


# Initialize MPI if not already
if (~MPI.Initialized())
  MPI.Init()
end
mpi_comm = MPI.COMM_WORLD # Communicator
mpi_rk = MPI.Comm_rank(mpi_comm) # Rank
mpi_sz = MPI.Comm_size(mpi_comm) # Size
MPI.Barrier(mpi_comm)

async_mpi=false; # prefer Non-blocking MPI communication


Np=Int(1e5) #Number of particles
Ns=1 #Number of species
dt=0.05
tmax=5


Nx1=1
Nx2=0
Nx3=0


mpi_Np=Int(Np/mpi_sz)
print(mpi_Np, " particles on proc", mpi_rk,"\n")



# Draw uniform
x1n,x2n,x3n,v1n,v2n,v3n=psmp.uniform_6d(mpi_Np,20 + mpi_rk*mpi_Np)
# sample testcase and retrieve default parameters
# params=psmp.weibel(x1n,x2n,x3n,v1n,v2n,v3n)
#params=psmp.weibel_streaming(x1n,x2n,x3n,v1n,v2n,v3n)
# params=psmp.landau(x1n,x2n,x3n,v1n,v2n,v3n,epsilon=0.5)
# params=psmp.jeans(x1n,x2n,x3n,v1n,v2n,v3n,epsilon=0.05)
params=psmp.KelvinHelmholtz(x1n,x2n,x3n,v1n,v2n,v3n,epsilon=0.05)


dt=params["dt"]
tmax=params["tmax"]
c=params["c"]
L=params["L"]
qn=ones(Float64,mpi_Np)*params["q"]
mn=ones(Float64,mpi_Np)*params["m"]

dt=0.1
tmax=500
# Likelihoods
fn=params["f0"](x1n,x2n,x3n,v1n,v2n,v3n)
gn=params["g0"](x1n,x2n,x3n,v1n,v2n,v3n)


import OrdinaryDiffEq: McAte2ConstantCache, McAte8ConstantCache
# Symmetric composition
dt_SS=McAte2ConstantCache(Float64,Float64) #
SS_len=Int(length(fieldnames(dt_SS))/2)
dtA=Array{Float64}(SS_len)
dtB=similar(dtA)
for n=1:SS_len
  eval(parse("dtA[$n]=dt_SS.a$n; dtB[$n]=dt_SS.b$n;"))
end
# dtA=0.5;dtB=0.5; # Strang splitting


#Particle mesh coupling
pm=pm_pif3d(Nx1,Nx2,Nx3,L...)
print("Total Fourier modes:", pm.N1*pm.N2*pm.N3, "\n")

# Fields
rho=Array{Complex{Float64}}(pm.N1,pm.N2,pm.N3)
Phi=Array{Complex{Float64}}(pm.N1,pm.N2,pm.N3)
E1=similar(Phi);E2=similar(Phi);E3=similar(Phi)
B1=similar(Phi);B2=similar(Phi);B3=similar(Phi)

# Weights
wn=fn./gn


Nt=Int(ceil(tmax/dt))
Epot=zeros(Float64,Nt,3)
Ekin=zeros(Float64,Nt,3)
Momentum=zeros(Float64,Nt,3)
kineticenergy=zeros(Float64,Nt)
ttime=collect((0:Nt-1)*dt)

#Initial poisson solve
rho=accum_osde(pm,x1n,x2n,x3n, wn.*qn)./Np
rho=MPI.allreduce(rho, MPI.SUM, mpi_comm)
solve_poisson!(pm,rho,Phi)
E1,E2,E3=gradient(pm,-Phi)
print(gauss_error(pm,rho,E1,E2,E3),"\n" )

# Initialize magnetic fields from initial condition
if haskey(params,"B")
   B1.=to_grid(pm,params["B"][1])
   B2.=to_grid(pm,params["B"][2])
   B3.=to_grid(pm,params["B"][3])
end



DTYPE=Float64
#Diagnostics
Nt=Int(ceil(tmax/dt))
Epot=zeros(DTYPE,Nt,3)
Bpot=zeros(DTYPE,Nt,3)
Ekin=zeros(DTYPE,Nt,3)
Momentum=zeros(DTYPE,Nt,3)
kineticenergy=zeros(DTYPE,Nt)
ttime=collect((0:Nt-1)*dt)


# dt_SS=McAte8ConstantCache(Float64,Float64) #
# SS_len=Int(length(fieldnames(dt_SS))/2)
# dtA2=Array{Float64}(SS_len)
# dtB2=similar(dtA2)
# for n=1:SS_len
#   eval(parse("dtA2[$n]=dt_SS.a$n; dtB2[$n]=dt_SS.b$n;"))
# end
using ProgressMeter

# function main_loop(x1n,x2n,x3n,v1n,v2n,v3n,mn,qn,wn,
#                    Ekin,Epot,Bpot,Momentum,
#                    pm,dt,dtA,dtB,Nt,c,
#                    mpi_rk,mpi_comm, async_mpi)
# Allocate temporary arrays
J1=zeros(Complex{Float64},pm.N1,pm.N2,pm.N3 )
J2=zeros(Complex{Float64},pm.N1,pm.N2,pm.N3 )
J3=zeros(Complex{Float64},pm.N1,pm.N2,pm.N3 )



using PyPlot


MPI.Barrier(mpi_comm)
if mpi_rk==0
  pg = Progress(Nt, 1,"Vlasov-Maxwell 6D")
end
tic()
for tdx=1:Nt
  if (tdx==3)
    tic()
  end
  if (mpi_rk==0)
    ProgressMeter.next!(pg)
  end

  Bpot[tdx,1]=L2norm(pm,B1)/2.
  Bpot[tdx,2]=L2norm(pm,B2)/2.
  Bpot[tdx,3]=L2norm(pm,B3)/2.
  Epot[tdx,1]=L2norm(pm,E1)/2.
  Epot[tdx,2]=L2norm(pm,E2)/2.
  Epot[tdx,3]=L2norm(pm,E3)/2.
  Ekin[tdx,1]=sum(v1n.^2.*wn)/Np/2.
  Ekin[tdx,2]=sum(v2n.^2.*wn)/Np/2.
  Ekin[tdx,3]=sum(v3n.^2.*wn)/Np/2.
  Momentum[tdx,1]=sum(v1n.*wn)/Np + (dot(pm,E2,B3)-dot(pm,E3,B2))
  Momentum[tdx,2]=sum(v2n.*wn)/Np + (dot(pm,E3,B1)-dot(pm,E1,B3))
  Momentum[tdx,3]=sum(v3n.*wn)/Np + (dot(pm,E1,B2)-dot(pm,E2,B1))

  if (mpi_rk==0)
    Momentum[tdx,1]+= (dot(pm,E2,B3)-dot(pm,E3,B2))
    Momentum[tdx,2]+= (dot(pm,E3,B1)-dot(pm,E1,B3))
    Momentum[tdx,3]+= (dot(pm,E1,B2)-dot(pm,E2,B1))
  end

 # integrate_H_E(pm,E1,E2,E3,B1,B2,B3,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,dt)
 # integrate_H_B(pm, E1,E2,E3,B1,B2,B3,c,dt)
 #  J1[:]=0.0;J2[:]=0.0;J3[:]=0.0 #mandatory
 # # Hamiltonian splitting
 # integrate_Hp(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
 #                   J1,J2,J3,B1,B2,B3,dt)
 #    J1[pm.N1n,pm.N2n,pm.N3n].=0
 #   J2[pm.N1n,pm.N2n,pm.N3n].=0
 #   J3[pm.N1n,pm.N2n,pm.N3n].=0
 #  E1.-=J1/Np;E2.-=J2/Np; E3.-=J3/Np
  J_req=Array{MPI.Request}(3)
  gdx=1;
  for gdx=1:length(dtA)

    if gdx==1
    integrate_H_E(pm,E1,E2,E3,B1,B2,B3,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,dtA[gdx]*dt)
    integrate_H_B(pm, E1,E2,E3,B1,B2,B3,c,dtA[gdx]*dt)
    end


   #J1_=similar(J1);J2_=similar(J2) ;J3_=similar(J3)
   J1[:]=0.0;J2[:]=0.0;J3[:]=0.0 #mandatory
   # Hamiltonian splitting
   # integrate_Hp_split_sym(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
   #                 J1,J2,J3,B1,B2,B3,[dtA[gdx]*dt],[dtB[gdx]*dt])

    # Exponential boris - not symplectic, but good for strong field
     integrate_Hp_boris_exp(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
                       J1,J2,J3,B1,B2,B3,dtA[gdx]*dt,dtB[gdx]*dt,async_mpi)

     # integrate_Hp_midpoint(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
     #                  J1,J2,J3,B1,B2,B3,dt2)

    # Subtract steady background for momentum conservation
    # J1[pm.N1n,:,:].=0
    # J2[:,pm.N2n,:].=0
    # J3[:,:,pm.N3n].=0
     #
     J1[pm.N1n,pm.N2n,pm.N3n].=0
     J2[pm.N1n,pm.N2n,pm.N3n].=0
     J3[pm.N1n,pm.N2n,pm.N3n].=0

     if async_mpi
      J_req[1],J1=MPI.iallreduce(J1, MPI.SUM, mpi_comm)
      J_req[2],J2=MPI.iallreduce(J2, MPI.SUM, mpi_comm)
      J_req[3],J3=MPI.iallreduce(J3, MPI.SUM, mpi_comm)

      #Only for boris, finish last step before communication
      integrate_vxB(pm, B1,B2,B3,x1n,x2n,x3n,v1n,v2n,v3n,
                                   qn, mn, dtB[gdx]*dt)
      else
        J1=MPI.allreduce(J1, MPI.SUM, mpi_comm)
        J2=MPI.allreduce(J2, MPI.SUM, mpi_comm)
        J3=MPI.allreduce(J3, MPI.SUM, mpi_comm)
    end

    if gdx==length(dtB)
      dt2=dtB[gdx]*dt
    else
      dt2=(dtA[gdx+1]+dtB[gdx])*dt
    end

      if async_mpi
      integrate_H_B(pm, E1,zeros(E2),zeros(E3),B1,B2,B3,c,dt2)
      MPI.Wait!(J_req[1])
      E1.-=J1/Np;
      integrate_H_E(pm,E1,x1n,x2n,x3n,v1n,qn,mn,dt2)


      integrate_H_B(pm, zeros(E1),E2,zeros(E3),B1,B2,B3,c,dt2)
      MPI.Wait!(J_req[2])
      E2.-=J2/Np;
      integrate_H_E(pm,E2,x1n,x2n,x3n,v2n,qn,mn,dt2)

      integrate_H_B(pm, zeros(E1),zeros(E2),E3,B1,B2,B3,c,dt2)
      MPI.Wait!(J_req[3])
      E3.-=J3/Np
      integrate_H_E(pm,E3,x1n,x2n,x3n,v3n,qn,mn,dt2)

      integrate_Faraday(pm,E1,E2,E3,B1,B2,B3,dt2)


    else
      integrate_H_B(pm, E1,E2,E3,B1,B2,B3,c,dt2)
      integrate_H_E(pm,E1,E2,E3,B1,B2,B3,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,dt2)
    end
  end


  x1n=mod.(x1n,pm.L1);  x2n=mod.(x2n,pm.L2);  x3n=mod.(x3n,pm.L3)



  if (mod(tdx-1,4)==0)
    if (tdx>2)  PyPlot.cla() end
    scatter(x1n,x2n,s=0.1)
   end


end
toc()

Ekin=MPI.allreduce(Ekin, MPI.SUM, mpi_comm)
Momentum=MPI.allreduce(Momentum, MPI.SUM, mpi_comm)



# end
#
# @profile main_loop(x1n,x2n,x3n,v1n,v2n,v3n,mn,qn,wn,
#                    Ekin,Epot,Bpot,Momentum,
#                    pm,dt,dtA,dtB,Nt,c,
#                    mpi_rk,mpi_comm, async_mpi)
# using ProfileView
# ProfileView.view()

# Profile.print(format=:tree,sortedby=:count)
# Profile.print(format=:flat,sortedby=:count)
# T=Float64;pdx=1;
#
# rho=accum_osde(pm,x1n,x2n,x3n, wn.*qn)./Np
# solve_poisson!(pm,rho,Phi)
# E1,E2,E3=gradient(pm,-Phi)
#
# J1[:]=0.0;J2[:]=0.0;J3[:]=0.0
#
 # integrate_Hp_midpoint(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
                #  J1,J2,J3,B1,B2,B3,dt)
# #J1[pm.N1n,:,:].=0; J2[:,pm.N2n,:].=0; J3[:,:,pm.N3n].=0
# J1[pm.N1n,pm.N2n,pm.N3n].=0
# J2[pm.N1n,pm.N2n,pm.N3n].=0
# J3[pm.N1n,pm.N2n,pm.N3n].=0
# E1.-=J1/Np;E2.-=J2/Np; E3.-=J3/Np
# #
# # rho=accum_osde(pm,x1n,x2n,x3n, wn.*qn)./Np
# # solve_poisson!(pm,rho*gamma,Phi)
# # E1_,E2_,E3_=gradient(pm,-Phi)
# #
# # print(E1-E1_, "\n" )
# # print(J1/Np,"\n")
rho=accum_osde(pm,x1n,x2n,x3n, wn.*qn)./Np
rho=MPI.allreduce(rho, MPI.SUM, mpi_comm)

print("Gauss Error: ",gauss_error(pm,rho,E1,E2,E3),"\n")
#########################################################
# Variance
# rho_var=accum_osde_var(pm,rho,x1n,x2n,x3n, wn.*qn)./Np
#
# rdx=sortperm(abs.(rho)[:])
#
# semilogy(abs.(rho[rdx]))
# plot(rho_var[rdx]/sqrt(Np))



# Only plot in repl
if (mpi_rk==0 && mpi_sz==1)

energy=sum(Epot,2)+sum(Ekin,2)+sum(Bpot,2)



# using Plots



figure()
semilogy(ttime,abs.((energy-energy[1])./energy[1]))
grid()
ylabel("rel. energy error")
xlabel("time ")

# plot(ttime, Ekin[:,1]+Epot[:,1])
#
figure()
 semilogy(ttime, Epot)
 figure()
 semilogy(ttime, Bpot)

# figure()
# semilogy(ttime, Bpot)

moment_error=abs.(Momentum[:,:].-reshape(Momentum[1,:],1,3));

figure()
semilogy(ttime,moment_error)
xlabel("time "); grid()
ylabel("absolute momentum error")


else
  ############# Finalize MPI
  MPI.Finalize()

end
