# 6D Vlasov - Poisson Single Species


# importall "./pif3d.jl
using Sobol
using ProgressMeter



Np=Int(1e4) #Number of particles
Ns=1 #Number of species
dt=0.05
tmax=5
gamma=1.0 # Gravity <0
q=-1.0 # Electrons
m=1.0

Nx1=1
Nx2=0
Nx3=0

# Default parameters
v01=0;v02=0;eps=0;delta=0;epsi=0; betai=0; betar=0;
sigma1=1;sigma2=1;sigma3=1;
c=1.0; k=0.5; eps=0; B0=zeros(3);

testcase="weibel"

#
if testcase=="weibel"
  eps=1e-3; # Amplitude of perturbation, 0.05 for linear, 0.5 for nonlinear
  betar=sign(q)*1e-3; betai=0;
  k=1.25;    # Wave vector
  sigma1=0.02/sqrt(2);
  sigma2=sqrt(12)*sigma1;
  sigma3=sigma2;
  v01=0;
  v02=0;
  delta=0;
  tmax=150;
  dt=0.2;
elseif testcase=="weibels"
  #Streaming Weibel instability
  sigma1=0.1/sqrt(2);
  sigma2=sigma1;
  k=0.2;
  betai=sign(q)*1e-3; betar=0;
  v01=0.5;
  v02=-0.1;
  delta=1/6.0;
  eps=0;
  tmax=150;
elseif testcase=="landau"
  c=1.0
  k=0.5; eps=0.5; B0=zeros(3);
  tmax=20

end


L1=2*pi/k
L2=2*pi/k
L3=2*pi/k


# Particles
x1n=zeros(Float64,Np)
x2n=zeros(Float64,Np)
x3n=zeros(Float64,Np)

v1n=zeros(Float64,Np)
v2n=zeros(Float64,Np)
v3n=zeros(Float64,Np)
#qn=q*ones(Float64,Np) # Charge
#mn=ones(Float64,Np) # Mass

fn=zeros(Float64,Np)
gn=zeros(Float64,Np)



# Initial condition for one species
f0(x1,x2,x3,v1,v2,v3)=((1+ eps*sin.(k*x1)).*
    exp.(- (v1.^2./sigma1^2 + v3.^2/sigma3^2)/2 ).*
    (delta*exp.(-(v2-v01).^2/2/sigma2.^2) +
    (1-delta)*exp.( - (v2-v02).^2/2/sigma2^2))*
    (2*pi)^(-1.5))/sigma3/sigma1/sigma2

g0(x1,x2,x3,v1,v2,v3)=exp.(-v1.^2./sigma1^2/2 -v3.^2./sigma3^2/2).*
    (delta*exp.(-(v2-v01).^2/2/sigma2.^2) +
    (1-delta)*exp.( - (v2-v02).^2/2/sigma2^2))*
    (2*pi)^(-1.5)/sigma3/sigma1/sigma2/L1/L2/L3;
# Fill with uniform numbers
using Sobol
sob = SobolSeq(6)
skip(sob, 4) # Skip some entries
for i=1:Np
  x1n[i],x2n[i],x3n[i],v1n[i],v2n[i],v3n[i]=Sobol.next(sob)
end

using SpecialFunctions
# Sampling
norminv=p->sqrt(2.).*erfinv.(2.*p - 1.0)
x1n*=L1;x2n*=L2;x3n*=L3;
v1n.=norminv(v1n)*sigma1
v2n.=norminv(v2n)*sigma2
v3n.=norminv(v3n)*sigma3
#Shifts
v2n[1:floor(Int,delta*Np)].+=v01
v2n[floor(Int,delta*Np)+1:end].+=v02
qn=ones(Float64,Np)*q
mn=ones(Float64,Np)*m


# Likelihoods
fn=f0(x1n,x2n,x3n,v1n,v2n,v3n)
gn=g0(x1n,x2n,x3n,v1n,v2n,v3n)



# Coefficients for Runge Kutta
comp_gamma=[1/(2-2^(1/3)), -2^(1/3)/(2-2^(1/3)),  1/(2-2^(1/3))]
comp_gamma=1.0


#Particle mesh coupling
pm=pif3d{Float64}(Nx1,Nx2,Nx3,L1,L2,L3)

print( pm.N1*pm.N2*pm.N3 )
# Fields
rho=Array{Complex{Float64}}(pm.N1,pm.N2,pm.N3)
Phi=Array{Complex{Float64}}(pm.N1,pm.N2,pm.N3)
E1=similar(Phi);E2=similar(Phi);E3=similar(Phi)
B1=similar(Phi);B2=similar(Phi);B3=similar(Phi)


E1n=zeros(Complex{Float64},Np)
E2n=zeros(Complex{Float64},Np)
E3n=zeros(Complex{Float64},Np)
wn=fn./gn


Nt=Int(ceil(tmax/dt))
Epot=zeros(Float64,Nt,3)
Ekin=zeros(Float64,Nt,3)
Momentum=zeros(Float64,Nt,3)
kineticenergy=zeros(Float64,Nt)
ttime=collect((0:Nt-1)*dt)

#Initial poisson solve
rho=accum_osde(pm,x1n,x2n,x3n, wn.*qn)./Np
solve_poisson!(pm,rho,Phi)
E1,E2,E3=gradient(pm,-Phi)
#Initial electric field
B1[:]=0;B2[:]=0;B3[:]=0
# Initial magnetic field
B3[pm.K1.==k,pm.K2.==0.,pm.K3.==0.]=betar + im*betai
B3[pm.K1.==-k,pm.K2.==0.,pm.K3.==0.]=betar - im*betai
B1[pm.N1n,pm.N2n,pm.N3n]=B0[1]
B2[pm.N1n,pm.N2n,pm.N3n]=B0[2]
B3[pm.N1n,pm.N2n,pm.N3n]=B0[3]

# rho[1,pm.Nx2+1,pm.Nx3+1]
print(gauss_error(pm,rho,E1,E2,E3),"\n" )

# Allocate temporary arrays
J1=zeros(Complex{Float64},pm.N1,pm.N2,pm.N3 )
J2=zeros(Complex{Float64},pm.N1,pm.N2,pm.N3 )
J3=zeros(Complex{Float64},pm.N1,pm.N2,pm.N3 )
dt2=dt

DTYPE=Float64
#Diagnostics
Nt=Int(ceil(tmax/dt))
Epot=zeros(DTYPE,Nt,3)
Bpot=zeros(DTYPE,Nt,3)
Ekin=zeros(DTYPE,Nt,3)
Momentum=zeros(DTYPE,Nt,3)
kineticenergy=zeros(DTYPE,Nt)
ttime=collect((0:Nt-1)*dt)
tic()
using ProgressMeter
# pg = Progress(10, 1,"Vlasov")
# for tdx=1:10
# sleep(0.1)
# ProgressMeter.next!(pg)
# end
# stop
#
# integrate_Hp_midpoint(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
#                 J1,J2,J3,B1,B2,B3,dt2)


# @time integrate_Hp_midpoint(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
#                  J1,J2,J3,B1,B2,B3,dt2)

# @time integrate_Hp_split_sym(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
#                                    J1,J2,J3,B1,B2,B3,dt2/2,dt2/2)
tic()
@showprogress 1 "Vlasov-Maxwell 6D" for tdx=1:Nt
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

  for gdx=1:length(comp_gamma)
     dt2=comp_gamma[gdx]*dt
  #
  #
    integrate_H_E(pm,E1,E2,E3,B1,B2,B3,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,dt2/2.)
    integrate_H_B(pm, E1,E2,E3,B1,B2,B3,c,dt2/2.)

    J1[:]=0.0;J2[:]=0.0;J3[:]=0.0 #mandatory
   # Hamiltonian splitting
   #integrate_Hp_split_sym(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
  #                   J1,J2,J3,B1,B2,B3,[dt2/2.],[dt2/2.])
   # Exponential boris - not symplectic, but good for strong field
    integrate_Hp_boris_exp(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,
                           J1,J2,J3,B1,B2,B3,[dt2/2],[dt2/2])

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
    E1.-=J1/Np;E2.-=J2/Np; E3.-=J3/Np


    integrate_H_B(pm, E1,E2,E3,B1,B2,B3,c,dt2/2.)
    integrate_H_E(pm,E1,E2,E3,B1,B2,B3,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,dt2/2.)

  end


  #x1n=mod.(x1n,L1);  x2n=mod.(x2n,L2);  x3n=mod.(x3n,L3)

end
toc()

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
print("Gauss Error: ",gauss_error(pm,rho,E1,E2,E3),"\n")


# Variance
# rho_var=accum_osde_var(pm,rho,x1n,x2n,x3n, wn.*qn)./Np
#
# rdx=sortperm(abs.(rho)[:])
#
# semilogy(abs.(rho[rdx]))
# plot(rho_var[rdx]/sqrt(Np))




energy=sum(Epot,2)+sum(Ekin,2)+sum(Bpot,2)

using PyPlot
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



# print(test_eval_basis(pm,10000))
#
# function push_Hp(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,dt,J1,J2,J3,B1,B2,B3)
#   Np=length(x1n)
#   @inbounds begin
#   for pdx=1:Np
#
#    eval_basis(pm,x1n[pdx],x2n[pdx],x3n[pdx],psin0 )
#   #  (x3n[pdx],v1n[pdx],v2n[pdx])=push_Hp3(pm, x1n[pdx],x2n[pdx],x3n[pdx],
#   #                         v1n[pdx],v2n[pdx],v3n[pdx],
#   #                          qn[pdx],mn[pdx],wn[pdx],
#   #                          psin0,dt/2,J3, B1,B2 )
#   #   (x2n[pdx],v1n[pdx],v3n[pdx])=push_Hp2(pm, x1n[pdx],x2n[pdx],x3n[pdx],
#   #                          v1n[pdx],v2n[pdx],v3n[pdx],
#   #                           qn[pdx],mn[pdx],wn[pdx],
#   #                           psin0,dt/2,J2, B1,B3 )
#      (x1n[pdx],v2n[pdx],v3n[pdx])=push_Hp1(pm, x1n[pdx],x2n[pdx],x3n[pdx],
#                 v1n[pdx],v2n[pdx],v3n[pdx],
#                  qn[pdx],mn[pdx],wn[pdx],
#                  psin0,dt,J1, B2,B3 )
#     # (x2n[pdx],v1n[pdx],v3n[pdx])=push_Hp2(pm, x1n[pdx],x2n[pdx],x3n[pdx],
#     #                        v1n[pdx],v2n[pdx],v3n[pdx],
#     #                         qn[pdx],mn[pdx],wn[pdx],
#     #                         psin0,dt/2,J2, B1,B3 )
#     # (x2n[pdx],v1n[pdx],v3n[pdx])=push_Hp2(pm, x1n[pdx],x2n[pdx],x3n[pdx],
#     #                        v1n[pdx],v2n[pdx],v3n[pdx],
#     #                         qn[pdx],mn[pdx],wn[pdx],
#     #                         psin0,dt/2,J2, B1,B3 )
#
#   end
#   end
#
#
#
# end
# using BenchmarkTools

#tic()
# push_Hp(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,dt,J1,J2,J3,B1,B2,B3)

# @benchmark integrate_Hp(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,J1,J2,J3,B1,B2,B3,dt)
# @time integrate_Hp(pm,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,wn,J1,J2,J3,B1,B2,B3,dt)

#
#
# maximum(abs(rho-rho2))
#
#  solve_poisson!(pm,rho*gamma,Phi)
#  E1_,E2_,E3_=gradient(pm,-Phi)
# print(gauss_error(pm,rho,E1_,E2_,E3_),"\n")



# print(E1)
# print(E1_)
#J1[1,:,:]=0.0
# J2[:,p,:]=0.0
# dE1[1,pm.Nx2+1,pm.Nx3+1]=0.0

 # (E1_-E1)[:,pm.Nx2+1,pm.Nx3+1]

 # DE=J1[:,pm.Nx2+1,pm.Nx3+1]/Np
# J1

# (dE1/Np/pm.L1/pm.L2/pm.L3)[:,pm.Nx2+1,pm.Nx3+1]


# end
# tic()
# for tdx=1:Nt
#
#   @inbounds for rkdx=1:length(rksd)
#
#
#     rho=accum_osde(pm,x1n,x2n,x3n, wn.*qn)./Np
#
#     solve_poisson!(pm,rho*gamma,Phi)
#
#     E1,E2,E3=gradient(pm,-Phi)
#
#     if rkdx==1
#
#     #fieldenergy[tdx]=H1seminorm(pm,Phi)/2.
#     #kineticenergy[tdx]=sum( (v1n.^2 .+v2n.^2.+v3n.^2).*wn)/Np/2.
#     Epot[tdx,1]=L2norm(pm,E1)/2.
#     Epot[tdx,2]=L2norm(pm,E2)/2.
#     Epot[tdx,3]=L2norm(pm,E3)/2.
#     Ekin[tdx,1]=sum(v1n.^2.*wn)/Np/2.
#     Ekin[tdx,2]=sum(v2n.^2.*wn)/Np/2.
#     Ekin[tdx,3]=sum(v3n.^2.*wn)/Np/2.
#     Momentum[tdx,1]=sum(v1n.*wn)/Np
#     Momentum[tdx,2]=sum(v2n.*wn)/Np
#     Momentum[tdx,3]=sum(v3n.*wn)/Np
#     end
#
#
#     eval_vectorfield(pm,x1n,x2n,x3n,E1,E2,E3,E1n,E2n,E3n)
#     v1n.+=dt*rksc[rkdx]*real(E1n).*(qn./mn)
#     v2n.+=dt*rksc[rkdx]*real(E2n).*(qn./mn)
#     v3n.+=dt*rksc[rkdx]*real(E3n).*(qn./mn)
#
#     # eval_scalar(pm,x1n,x2n,x3n,E1,En)
#     # v1n.+=dt*rksc[rkdx]*real(En)
#     # eval_scalar(pm,x1n,x2n,x3n,E2,En)
#     # v2n.+=dt*rksc[rkdx]*real(En)
#     # eval_scalar(pm,x1n,x2n,x3n,E3,En)
#     # v3n.+=dt*rksc[rkdx]*real(En)
#
#     # v1n+=dt*rksc[rkdx]*eval_scalar(pm,x1n,x2n,x3n,E1*q/m)
#     # v2n+=dt*rksc[rkdx]*eval_scalar(pm,x1n,x2n,x3n,E2*q/m)
#     # v3n+=dt*rksc[rkdx]*eval_scalar(pm,x1n,x2n,x3n,E3*q/m)
#
#     x1n.+=dt*rksd[rkdx].*v1n
#     x2n.+=dt*rksd[rkdx].*v2n
#     x3n.+=dt*rksd[rkdx].*v3n
#
#
#   end
#   x1n=mod(x1n,L1)
#   x2n=mod(x2n,L2)
#   x3n=mod(x3n,L3)
#
# end
# toc()
#
# using PyPlot
# # semilogy(ttime,Epot)
#
#
# energy=gamma*sum(Epot,2)+sum(Ekin,2)
# moment_error=sum(abs(Momentum[:,:].-reshape(Momentum[1,:],1,3)),2);
#
# # E1n_=zeros(Float64,Np)
# # using BenchmarkTools
# #
# # @benchmark eval_scalar(pm,x1n,x2n,x3n,E1*q/m,E1n)
# # @benchmark eval_scalar2(pm,x1n,x2n,x3n,E1*q/m,E1n_)
# # print((E1n_[1]-real(E1n[1]))/real(E1n[1]) )
# figure()
# semilogy(ttime,abs((energy-energy[1])./energy[1]))
# grid()
# ylabel("rel. energy error")
# xlabel("time ")
# # savefig("$prefix\_energy_error.png")
# figure()
# semilogy(ttime,moment_error)
# xlabel("time "); grid()
# ylabel("absolute momentum error")
# # savefig("$prefix\_momentum_error.png")
#
#
# # figure()
# # plot(energy)
# #
# #
# # figure()
# # semilogy(moment_error)
# #
# # grid()
# #
# #
# figure()
# plot(abs(Phi[:]))
# # using PyPlot
# (L2norm(pm,E1)+L2norm(pm,E2)+L2norm(pm,E3))/2
#
# fieldenergy[end]
# Nviz=Np
# figure()
# sc=scatter3D(x1n[1:Nviz],x2n[1:Nviz],x3n[1:Nviz],s=2,c=fn[1:Nviz],depthshade="True",lw=0)
# colorbar(sc)
# tight()
# plot(fieldenergy)
#
# scatter(x1n,x2n,s=3,lw=0)
#
# figure()
# semilogy(ttime,fieldenergy)
# fig=figure()
# ax = gca(projection="3d")
#
# ax[:scatter](x1n,x2n,x3n)
# @time eval_scalar(pm,x1n,x2n,x3n,E1*q/m,En)

# figure()
# plot(x1n,x2n,x3n)

#@time rho=rhs_particle(pm,x1n,x2n,x3n, wn)./Np*q


#Damping rate for linear Landau
#omega=1.415661888604536 - 0.153359466909605im;
#semilogy(ttime[ttime.<10.],0.5*fieldenergy[1]*abs(exp(-im*omega*(ttime[ttime.<10.]-0.4))).^2)


#plot(abs(rho)[:])

#rho

# plot(abs(Phi[:]))


# type vlasov6d
#
#   Np::Int #Number of particles
#   Ns::Int #Number of species
#
#   x1n::Array{Float64,1}
#   x2n::Array{Float64,1}
#   x3n::Array{Float64,1}
#
#   v1n::Array{Float64,1}
#   v2n::Array{Float64,1}
#   v3n::Array{Float64,1}
#   qn::Array{Float64,1} # Charge
#   mn::Array{Float64,1} # Mass
#
#   fn::Array{Float64,1}
#   gn::Array{Float64,1}
#
#   function vlasov6d(Np,Ns)
#     x=zeros(Float64,Int(Np)*Int(Ns))
#     new(Int(Np),Int(Ns),x,x,x, x,x,x, x,x,x,x)
#
#   end
# end
#
#
# function std_Maxwellian(x1,x2,x3,v1,v2,v3)
#   return exp(-(v1.^2+v2.^2+v3.^2)/2 )/(sqrt(2*pi))^3
# end
#
#
#
# function sample_std_Maxwellian(sim::vlasov6d, L1,L2,L3   )
#   sim.x1n=rand(Float64,sim.Np)*L1
#   sim.x2n=rand(Float64,sim.Np)*L2
#   sim.x3n=rand(Float64,sim.Np)*L3
#
#   sim.v1n=randn(Float64,sim.Np)
#   sim.v2n=randn(Float64,sim.Np)
#   sim.v3n=randn(Float64,sim.Np)
#
#   gn=std_Maxwellian(sim.x1n,sim.x2n,sim.x3n,sim.v1n,sim.v2n,sim.v3n)/L1/L2/L3
# end
#
#
# f0=(x1,x2,x3,v1,v2,v3)->(exp(-(v1.^2+v2.^2+v3.^2)/2 )/(sqrt(2*pi))^3
#
#
# #Electron only
# sim=vlasov6d(1e5,1)
# k=0.5;
# epse=0.1
# #Particle mesh coupling
# pm=pif3d(2,2,2,2*pi/k)
#
# sim.qn[:]=1.;sim.mn[:]=1.
# sample_std_Maxwellian(sim,pm.L1,pm.L2,pm.L3)
# sim.fn=(1+epse*cos(k*x)  std_Maxwellian(sim.x1n,sim.x2n,sim.x3n,sim.v1n,sim.v2n,sim.v3n)
#
# rhs=rhs.particle(sim.x1n,sim.x2n,sim.x3n, sim.fn/sim.gn) /Np*Ns
# sim.x1n


# using BenchmarkTools
# @benchmark rho=accum_osde(pm,x1n,x2n,x3n, wn.*qn)./Np
# @benchmark rho=accum_osde2(pm,x1n,x2n,x3n, wn.*qn)./Np
