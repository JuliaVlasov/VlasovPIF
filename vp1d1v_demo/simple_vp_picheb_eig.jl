using FastTransforms: cheb2leg, leg2cheb
using ApproxFun
#using Plots
Np=Int(1e5)
Nx=Int(25)
k0=0.5
dt=0.1
tmax=8
L=2*pi/k0
deg=0; #Smoothness of periodic boundary


function f(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.05.*cos(k0*x))
end

xk=rand(Np)*L
vk=randn(Np)
fk=f(xk,vk)
wk=f(xk,vk)*L

# Derivative of a chebyshev series
function chebTdx_series{T}(x::T, c::Array{T,1})
  N=length(c)-1;
  b1::T=0
  b2::T=0
  @inbounds for n=N:-1:2
     @inbounds b0::T=c[n+1]+ 2*x*(n+1)/n*b1 - (n+2)/n*b2
    b2=b1; b1=b0;
  end
  return c[2]+ 4*x.*b1 - 3*b2
end
# Derivative of a chebyshev series
function chebTdx_series{T}(x::Array{T,1}, c::Array{T,1})
  N=length(c)-1;
  alpha=Array(T,N-1)
  beta=Array(T,N-1)
  for n=N:-1:2
    alpha[n-1]=2*(n+1)/n
    beta[n-1]=-(n+2)/n
  end
  E=Array(T,length(x))
   @inbounds for pdx=1:length(x)
   b1::T=0
   b2::T=0
   @inbounds for n=N:-1:2
      @inbounds b0::T=c[n+1]+ x[pdx]*alpha[n-1]*b1 + beta[n-1]*b2
     b2=b1; b1=b0;
    end
    E[pdx]=c[2]+ 4*x[pdx].*b1 - 3*b2
  end
  return E
end
# Legendre polynomials
function LegendreP{T}(x::T,N::Int)
  Pn=Array(T, N)
  Pn[1]=1
  Pn[2]=x
  @inbounds for idx=3:N
    n=idx-2
    Pn[idx]= x.*(2*n+1)/(n+1).*Pn[idx-1]- n/(n+1).*Pn[idx-2]
  end
  return Pn
end

@inbounds function LegendreP_series{T}(x::Array{T,1}, u::Array{T,1})
  N::Int=length(u)-1;
  alpha=Array(T,N+1)
  beta=Array(T,N+1)
  #
  for n=0:N
     alpha[n+1]=(2*n+1)/(n+1)
     beta[n+1]=-(n+1)/(n+2)
  end
  E=Array(T,length(x))
   for pdx=1:length(x)
   b1::T=0
   b2::T=0
   for n=N:-1:0
     b0::T=u[n+1]+ x[pdx]*alpha[n+1]*b1 + beta[n+1]*b2
     b2=b1; b1=b0;
    end
    E[pdx]=b1
  end
  return E
end



# #@time LegendreP(xk[1],32)
# function  rhs_particle_legendre{T}(xk::Array{T,1},wk::Array{T,1}, Nx::Int)
#     Np=length(xk)
#     rho=zeros(T,Nx)
#
#     @inbounds for pdx=1:Np
#       P0::T=1.0; P1::T=xk[pdx];
#       rho[1]+=P0.*wk[pdx]
#       rho[2]+=P1.*wk[pdx]
#       @inbounds for idx=3:Nx
#         n=idx-2
#         P2::T=xk[idx].*(2*n+1)/(n+1).*P1- n/(n+1).*P0
#         P0=P1; P1=P2;
#         rho[idx]+=P2.*wk[pdx]
#       end
#     end
#   return rho
# end
function  rhs_particle_legendre{T}(xk::Array{T,1},wk::Array{T,1}, Nx::Int)
    const Np=length(xk)
    rho=zeros(T,Nx)
    # Precalculate coefficients
    alpha=Array(T,Nx-2)
    beta=Array(T,Nx-2)
    @inbounds for n=1:Nx-2
      alpha[n]=(2*n+1)/(n+1)
      beta[n]=-n/(n+1)
    end

    @inbounds for pdx=1:Np
      P0::T=1; P1::T=xk[pdx];
      rho[1]+=P0.*wk[pdx]
      rho[2]+=P1.*wk[pdx]
      @inbounds for idx=1:Nx-2
        P2::T=xk[pdx].*alpha[idx].*P1+ beta[idx].*P0
        P0=P1; P1=P2;
        @inbounds rho[idx+2]+=P2.*wk[pdx]
      end
    end
  return rho
end
function  rhs_particle{T}(xk::Array{T,1},wk::Array{T,1}, Nx::Int)
    Np=length(xk)
    rho=zeros(T,Nx)
    @inbounds for pdx=1:Np
      rho+=LegendreP(xk[pdx],Nx).*wk[pdx]
    end
  return rho
end



OM=Interval(0,L) # Domain
S=ApproxFun.Space(OM) #Default chebyshev space
B=periodic(S,deg) # Periodic boundary
B2=DefiniteIntegral(S) # Nullspace condition
D2=-Derivative(S,2) #Laplace operator
D1=Derivative(S,1) # First derivative for H1 norm
IN=DefiniteIntegral(S) # First derivative for L2 norm

#Phi = \([B;B2;D2],[0.0;0.0;rho];tolerance=1E-16,maxlength=Nx+2)
QR = qrfact([B;B2;D2])

# Coefficients for Runge Kutta
rksd=[2/3.0, -2/3.0, 1  ]
rksc=[ 7/24.0, 3/4.0, -1/24.0]

# Phi=QR \[zeros(Float64,deg); 0.0;0.0;rho]

DE, DV = eigs([B;B2;D2],Nx)
x=collect(linspace(0.0,L))
for idx=1:length(DV)
  DV[idx].coefficients=DV[idx].coefficients[1:Nx]
  DV[idx].coefficients[1]=0
  DV[idx].coefficients=DV[idx].coefficients./norm(DV[idx].coefficients)
  if DE[idx]==0
    DE[idx]=Inf
  end
  #plot(x,map(x->DV[idx](x),x))
end
Nf=2
p = sortperm(abs(DE));DV=DV[p[1:Nf]]; DE=DE[p[1:Nf]]



Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt)
kineticenergy=zeros(Float64,Nt)
momentum=zeros(Float64,Nt)
rho=Fun(S)
E=Fun(S)
Phi=Fun(S,zeros(Float64,Nx))





tic()
@inbounds for tdx=1:Nt

  @inbounds for rkdx=1:length(rksd)

    # Bin to right hand side
    #rhs=rhs_particle( xk/L*2-1,wk,Nx)/Np*(2/L)
    rhs=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np*(2/L)
    # Legendre L^2 projection
    rhs=rhs./(2./(2*(0:Nx-1)+1.0))
    # Remove ion background (quick'n dirty)
    rhs[1]=0
    #Transform to Chebyshev
    rhs=leg2cheb(rhs)

    #Eigenvalue filter
    Phi.coefficients=zeros(Float64,Nx)
    for idx=1:length(DV)
    Phi.coefficients.+=(dot(rhs,DV[idx].coefficients)/DE[idx]).*DV[idx].coefficients
    end


        # Solve periodic Poisson equation with Chebyshev
        # Chebyshev periodic Poisson
        #rho.coefficients=rhs
        # Phi = \([B;B2;D2],[0.0;0.0;rho];tolerance=1E-16)
        #Phi=QR \[zeros(Float64,deg);0.0;0.0;rho]



    #E=-rho./(-im*kx)
    if rkdx==1
      fieldenergy[tdx]=coefficients(IN*(D1*Phi)^2)[1]/2
      #*(L/2)
      #fieldenergy[tdx]=dot(Phi.coefficients[1:Nx],rhs)/2*(L/2)/(2/L)
      kineticenergy[tdx]=0.5.*sum(vk.^2.*wk )/Np
      momentum[tdx]=sum(vk.*wk)/Np
    end
    E=Fun(-D1*Phi,S)
    #E.coefficients[1]=0
    E_leg=cheb2leg(E.coefficients)
    vk.+=dt*rksc[rkdx]*LegendreP_series(xk/L*2-1,E_leg)
    xk.+=dt*rksd[rkdx].*vk
    xk=mod(xk,L)
  end

  #print(tdx)
end
toc()
ttime=(0:Nt-1)*dt

#print(ttime)
#plot(Phi)
using PyPlot
#: semilogy,plot,figure,ylabel,xlabel,grid,savefig


prefix="landau_strong"
prefix="legendre"


# figure()
# title("magnitude of coefficients")
# semilogy(abs(coefficients(rho)),label=L"density $\rho$")
# semilogy(abs(coefficients(Phi)),label=L"potential $\Phi$")
# semilogy(abs(coefficients(D1*Phi)),label=L"field $E$")
# xlabel("index")
# legend()
# grid()
# savefig("$prefix\_field_coeffs.png")
# figure()
# semilogy(ttime,kineticenergy)

figure()
semilogy(ttime,fieldenergy)
omega=1.415661888604536 - 0.153359466909605im;
semilogy(ttime[ttime.<8.],
0.5*fieldenergy[1]*abs(exp(-im*omega*(ttime[ttime.<8.]-0.4))).^2,
lw=3,"r")
grid();autoscale(tight="True")
ylabel("electrostatic energy");grid()
xlabel("time")
#tight_layout()
savefig("$prefix\_fieldenergy.png")


figure()
semilogy(ttime,kineticenergy)
grid();
#autoscale(tight="True")
ylabel("kinetic energy")
xlabel("time")
savefig("$prefix\_kineticenergy.png")

figure()
energy=kineticenergy+fieldenergy
semilogy(ttime,abs((energy-energy[1])./energy[1]))
grid()
ylabel("rel. energy error")
xlabel("time ")
savefig("$prefix\_energy_error.png")
figure()
semilogy(ttime,abs(momentum-momentum[1]))
xlabel("time "); grid()
ylabel("absolute momentum error")
savefig("$prefix\_momentum_error.png")


figure()
scatter(xk,vk,s=1, c=fk,lw=0)
#autoscale(tight="True")
axis([0,L,minimum(vk),maximum(vk)])
colorbar();tight_layout()
xlabel("x");ylabel("v")
savefig("$prefix\_particles.png")

# QR*Phi
#Phi
function  rhs_var_particle_legendre{T}(xk::Array{T,1},wk::Array{T,1},
               Nx::Int,   rhs::Array{T,1})
    const Np=length(xk)
    rho=zeros(T,Nx)
    # Precalculate coefficients
    alpha=Array(T,Nx-2)
    beta=Array(T,Nx-2)
    @inbounds for n=1:Nx-2
      alpha[n]=(2*n+1)/(n+1)
      beta[n]=-n/(n+1)
    end

    @inbounds for pdx=1:Np
      P0::T=1; P1::T=xk[pdx];
      rho[1]+=(P0.*wk[pdx]-rhs[1]).^2
      rho[2]+=(P1.*wk[pdx]-rhs[2]).^2
      @inbounds for idx=1:Nx-2
        P2::T=xk[pdx].*alpha[idx].*P1+ beta[idx].*P0
        P0=P1; P1=P2;
        @inbounds rho[idx+2]+=(P2.*wk[pdx]-rhs[idx+2]).^2
      end
    end
  return rho
end

rhs=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np
rhs_var=rhs_var_particle_legendre( xk/L*2-1,wk,Nx,rhs)/(Np-1)

# Legendre L^2 projection
rhs=rhs./(2./(2*(0:Nx-1)+1.0))*(2/L)
rhs_var=rhs_var./(2./(2*(0:Nx-1)+1.0)).^2*(2/L).^2
rhs[1]=0

figure()
title("Legendre coefficients")
semilogy(abs(rhs),label="magnitude")
plot(sqrt(rhs_var/Np),label="std. deviation")
grid();autoscale(tight="True")
xlabel("index")
legend()
savefig("$prefix\_legendre_coeffs_var.png")



#using JLD
#@save "$prefix\_results.jld"


#
 # figure()
 # errorbar(1:Nx, rhs, yerr=sqrt(rhs_var/Np) , fmt="-o")

#
# rhs=rho.coefficients
#
#
#
# Phi=QR \[0.0;0.0;rho]
#
#
#  coefficients(DV[1])
# #
#
# #
# # using Plots: plot
# #
#
# plot()
# plot(x,map(x->DV[1](x),x))

# coefficients(V[1:2])

# figure()
# semilogy(ttime,fieldenergy)
# grid()
# xlabel("time")
# ylabel("electrostatic energy")
#using Plots
#plot(Phi)
