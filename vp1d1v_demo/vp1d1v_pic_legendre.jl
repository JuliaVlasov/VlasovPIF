using PyPlot

Np=Int(1e6)
Nx=Int(20)
k0=0.5
dt=0.01
tmax=25
L=2*pi/k0*2
deg=0; #Smoothness of periodic boundary


function f(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.05*sin(k0*x))
end
function g(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi)/L
end



xk=rand(Np)*L
vk=randn(Np)
fk=f(xk,vk)
wk=f(xk,vk)./g(xk,vk)

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

# Coefficients for Runge Kutta
rksd=[2/3.0, -2/3.0, 1  ]
rksc=[ 7/24.0, 3/4.0, -1/24.0]

# Phi=QR \[zeros(Float64,deg); 0.0;0.0;rho]



Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt)
kineticenergy=zeros(Float64,Nt)
momentum=zeros(Float64,Nt)
rho=zeros(Float64,Nx)
Phi=zeros(Float64,Nx-2)
E=zeros(Float64,Nx)

tic()
@inbounds for tdx=1:Nt

  @inbounds for rkdx=1:length(rksd)

    #outdx=(xk.<=0) | (xk.>=L)
    indx=(xk.>0) & (xk.<L)
    #xk[outdx ]=0.
    #wk2=wk
    #wk2[outdx]=0.

    # Bin to right hand side
    rhs=rhs_particle_legendre( xk[indx]/L*2-1,wk[indx],Nx)/Np
    # Legendre L^2 projection
    rho=rhs./(2./(2*(0:Nx-1)+1.0))*(2/L)
    # Remove ion background (quick'n dirty)
    rhs[1].-=1.0*L


    Phi.=-(rhs[1:Nx-2]-rhs[3:Nx])./sqrt(4*(0:Nx-3)+6)
    E[2:Nx-1]=-Phi.*sqrt( (0:Nx-3) + 3/2)

    if rkdx==1
      fieldenergy[tdx]=sum(E.^2.*(2./(2*(0:Nx-1)+1.0)))/2*(L/2)
      kineticenergy[tdx]=0.5.*sum(vk.^2.*wk )/Np
      momentum[tdx]=sum(vk.*wk)/Np
    end

    vk[indx].+=dt*rksc[rkdx]*LegendreP_series(xk[indx]/L*2-1,E)
    xk.+=dt*rksd[rkdx].*vk
    xk=mod(xk,L)
    # Reflecting boundary
  end

  #print(tdx)
end
toc()
ttime=(0:Nt-1)*dt

#print(ttime)
#plot(Phi)
#: semilogy,plot,figure,ylabel,xlabel,grid,savefig


prefix="plots/legendre_landau_weak"
#prefix="legendre"


# figure()
# title("magnitude of coefficients")
# semilogy(abs(coefficients(rho)),label=L"density $\rho$")
# semilogy(abs(coefficients(Phi)),label=L"potential $\Phi$")
# semilogy(abs(coefficients(D1*Phi)),label=L"field $E$")
# xlabel("index")
# legend()
# grid()
# savefig("$prefix\_field_coeffs.png")


PyPlot.matplotlib[:rc]("font", size=18)

figure()
semilogy(ttime,fieldenergy)
omega=1.415661888604536 - 0.153359466909605im;
semilogy(ttime[ttime.<8.],
0.5*fieldenergy[1]*abs(exp(-im*omega*(ttime[ttime.<8.]-0.4))).^2,
lw=3,"r")
autoscale(tight="True");grid()
ylabel("electrostatic energy");
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
#axis([0,L,minimum(vk),maximum(vk)])
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
end#



#
# figure()
# semilogy(ttime,fieldenergy)
# grid()
# xlabel("time")
# ylabel("electrostatic energy")





#
# rhs=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np
# rhs_var=rhs_var_particle_legendre( xk/L*2-1,wk,Nx,rhs)/(Np-1)
# #
# # # Legendre L^2 projection
# rhs=rhs./(2./(2*(0:Nx-1)+1.0))*(2/L)
# rhs_var=rhs_var./(2./(2*(0:Nx-1)+1.0)).^2*(2/L).^2
# rhs[1]=0
# #
# figure()
# title("Legendre coefficients")
# semilogy(abs(rhs),label="magnitude")
# plot(sqrt(rhs_var/Np),label="std. deviation")
# grid();autoscale(tight="True")
# xlabel("index")
# legend()
# savefig("$prefix\_legendre_coeffs_var.png")



# using JLD
# @save "$prefix\_results.jld"


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

#using Plots
#plot(Phi)
