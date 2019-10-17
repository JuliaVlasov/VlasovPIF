# 6D Vlasov - Poisson Single Species

using Sobol

DTYPE=Float64

Np=Int(1e4) #Number of particles
Ns=1 #Number of species
dt=0.05
tmax=20
gamma=1.0 # Gravity <0
q=-1.0 # Electrons
m=1.0

Nx1=1
Nx2=1
Nx3=1

# Default parameters
v01=0;v02=0;eps=0;delta=0;epsi=0; betai=0; betar=0;
sigma1=1;sigma2=1;sigma3=1;

k=0.5; eps=0.1;



L1=DTYPE(2*pi/k)
L2=DTYPE(2*pi/k)
L3=DTYPE(2*pi/k)


# Particles
x1n=zeros(DTYPE,Np)
x2n=zeros(DTYPE,Np)
x3n=zeros(DTYPE,Np)

v1n=zeros(DTYPE,Np)
v2n=zeros(DTYPE,Np)
v3n=zeros(DTYPE,Np)
#qn=q*ones(Float64,Np) # Charge
#mn=ones(Float64,Np) # Mass

fn=zeros(DTYPE,Np)
gn=zeros(DTYPE,Np)



# Initial condition
f0(x1,x2,x3,v1,v2,v3)=((1+ eps*sin.(k*x1+pi/5)).*
    exp.(- (v1.^2./sigma1^2 + v3.^2/sigma3^2)/2 ).*
    (delta*exp.(-(v2-v01).^2/2/sigma2.^2) +
    (1-delta)*exp.( - (v2-v02).^2/2/sigma2^2))
    *(2*pi)^(-1.5))/sigma3/sigma1/sigma2

g0(x1,x2,x3,v1,v2,v3)=(exp.(-v1.^2./sigma1^2/2 -v3.^2./sigma1^2/2).*
    (delta*exp.(-(v2-v01).^2/2/sigma2.^2) +
    (1-delta)*exp.( - (v2-v02).^2/2/sigma2^2))
    *(2*pi)^(-1.5) )/sigma3/sigma1/sigma2/L1/L2/L3;
# Fill with uniform numbers
using Sobol
sob = SobolSeq(6)
skip(sob, 22) # Skip some entries
for i=1:Np
  x1n[i],x2n[i],x3n[i],v1n[i],v2n[i],v3n[i]=next(sob)
end

# Sampling
using SpecialFunctions

norminv(p)=sqrt(2).*erfinv.(2 *p - 1.0)
v1n=norminv(v1n)*sigma1
v2n=norminv(v2n)*sigma2;
v3n=norminv(v3n)*sigma3;
x1n=x1n*L1;x2n=x2n*L2;x3n=x3n*L3;
qn=q*ones(DTYPE,Np)
mn=ones(DTYPE,Np)*m


# Likelihoods
fn=f0(x1n,x2n,x3n,v1n,v2n,v3n)
gn=g0(x1n,x2n,x3n,v1n,v2n,v3n)



#using PyPlot
# figure()
#h = plt[:hist](v1n,v40)


# Coefficients for Runge Kutta
const rksd=[2/3.0, -2/3.0, 1  ]
const rksc=[ 7/24.0, 3/4.0, -1/24.0]

#Particle mesh coupling
pm=pm_pif3d(Nx1,Nx2,Nx3,L1,L2,L3)


print( pm.N1*pm.N2*pm.N3 )
# Fields
rho=Array{Complex{DTYPE}}(pm.N1,pm.N2,pm.N3)
Phi=Array{Complex{DTYPE}}(pm.N1,pm.N2,pm.N3)
E1=Array(Phi);E2=Array(Phi);E3=Array(Phi)
E1n=zeros(Complex{DTYPE},Np)
E2n=zeros(Complex{DTYPE},Np)
E3n=zeros(Complex{DTYPE},Np)
wn=fn./gn




Nt=Int(ceil(tmax/dt))
Epot=zeros(DTYPE,Nt,3)
Ekin=zeros(DTYPE,Nt,3)
Momentum=zeros(DTYPE,Nt,3)
kineticenergy=zeros(DTYPE,Nt)
ttime=collect((0:Nt-1)*dt)
tic()
for tdx=1:Nt

  @inbounds for rkdx=1:length(rksd)
    rho=accum_osde(pm,x1n,x2n,x3n, wn.*qn)./Np

    solve_poisson!(pm,rho*gamma,Phi)

    E1,E2,E3=gradient(pm,-Phi)

    if rkdx==1

    #fieldenergy[tdx]=H1seminorm(pm,Phi)/2.
    #kineticenergy[tdx]=sum( (v1n.^2 .+v2n.^2.+v3n.^2).*wn)/Np/2.
    Epot[tdx,1]=L2norm(pm,E1)/2.
    Epot[tdx,2]=L2norm(pm,E2)/2.
    Epot[tdx,3]=L2norm(pm,E3)/2.
    Ekin[tdx,1]=sum(v1n.^2.*wn)/Np/2.
    Ekin[tdx,2]=sum(v2n.^2.*wn)/Np/2.
    Ekin[tdx,3]=sum(v3n.^2.*wn)/Np/2.
    Momentum[tdx,1]=sum(v1n.*wn)/Np
    Momentum[tdx,2]=sum(v2n.*wn)/Np
    Momentum[tdx,3]=sum(v3n.*wn)/Np
    end


    # eval_vectorfield(pm,x1n,x2n,x3n,E1,E2,E3,E1n,E2n,E3n)
    #eval_scalar(pm,x1n,x2n,x3n,[E1,E2,E3],[E1n,E2n,E3n])
    #  v1n.+=(dt*rksc[rkdx]).*real(E1n).*(qn./mn)
    #  v2n.+=(dt*rksc[rkdx])*real(E2n).*(qn./mn)
    #  v3n.+=(dt*rksc[rkdx])*real(E3n).*(qn./mn)

 integrate_H_E(pm,E1,E2,E3,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,DTYPE(dt*rksc[rkdx]))
    #  eval_scalar(pm,x1n,x2n,x3n,E1,En)
    # v1n.+=dt*rksc[rkdx]*real(En).*(qn./mn)
    # eval_scalar(pm,x1n,x2n,x3n,E2,En)
    # v2n.+=dt*rksc[rkdx]*real(En).*(qn./mn)
    # eval_scalar(pm,x1n,x2n,x3n,E3,En)
    # v3n.+=dt*rksc[rkdx]*real(En).*(qn./mn)

    # v1n+=dt*rksc[rkdx]*eval_scalar(pm,x1n,x2n,x3n,E1*q/m)
    # v2n+=dt*rksc[rkdx]*eval_scalar(pm,x1n,x2n,x3n,E2*q/m)
    # v3n+=dt*rksc[rkdx]*eval_scalar(pm,x1n,x2n,x3n,E3*q/m)

    x1n.+=dt*rksd[rkdx].*v1n
    x2n.+=dt*rksd[rkdx].*v2n
    x3n.+=dt*rksd[rkdx].*v3n


  end
  x1n=mod(x1n,L1)
  x2n=mod(x2n,L2)
  x3n=mod(x3n,L3)

end
toc()




using PyPlot
# semilogy(ttime,Epot)


energy=gamma*sum(Epot,2)+sum(Ekin,2)
moment_error=sum(abs(Momentum[:,:].-reshape(Momentum[1,:],1,3)),2);

# E1n_=zeros(Float64,Np)
# using BenchmarkTools
#
# @benchmark eval_scalar(pm,x1n,x2n,x3n,E1*q/m,E1n)
# @benchmark eval_scalar2(pm,x1n,x2n,x3n,E1*q/m,E1n_)
# print((E1n_[1]-real(E1n[1]))/real(E1n[1]) )
figure()
semilogy(ttime,abs.((energy-energy[1])./energy[1]))
grid()
ylabel("rel. energy error")
xlabel("time ")
# savefig("$prefix\_energy_error.png")
figure()
semilogy(ttime,moment_error)
xlabel("time "); grid()
ylabel("absolute momentum error")
# savefig("$prefix\_momentum_error.png")

#
figure()
 semilogy(ttime,Epot)

# figure()
# plot(energy)
#
#
# figure()
# semilogy(moment_error)
#
# grid()
#
#
# figure()
# plot(abs(Phi[:]))
# # using PyPlot
# # (L2norm(pm,E1)+L2norm(pm,E2)+L2norm(pm,E3))/2
# #
# # fieldenergy[end]
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
