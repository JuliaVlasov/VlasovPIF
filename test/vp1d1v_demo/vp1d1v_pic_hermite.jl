using PyPlot
using SpecialFunctions
Np=Int(1e4)
Nx=Int(6)
dt=0.05
tmax=1000
deg=0; #Smoothness of periodic boundary
sigmaL=4*pi;
vth=0.1;
gamma=1; # Electrons (-1), gravity 1
eps=0
pad=0

function f(x,v)
  return exp.(-0.5.*(v/vth).^2)/sqrt(2*pi)/vth.*
              exp.(-0.5.*(x/sigmaL).^2).*(1+ eps*sqrt(2)*((x/sigmaL).^2-0.5) )/sqrt(2*pi)/(1+0.5*eps)
end
function g(x,v)
  return exp.(-0.5.*(v/vth).^2)./sqrt(2*pi)/vth.*exp.(-0.5.*(x/sigmaL).^2)/sqrt(2*pi)/sigmaL
end

using Sobol
sob = SobolSeq(2)
skip(sob, 5) # Skip some entries
xk=Array(Float64,Np); vk=similar(xk)
for i=1:Np
  xk[i],vk[i]=Sobol.next(sob)
end

# Sampling
norminv=p->sqrt(2.).*erfinv.(2.*p - 1.0)
xk.=norminv(xk)*sigmaL
vk.=norminv(vk)*vth

fk=f(xk,vk)
wk=f(xk,vk)./g(xk,vk)



# Hermite functions
function HermiteH{T}(x::T,N::Int)
  Hn=Array(T, N)
  Hn[1]=1.
  Hn[2]=sqrt(2)*x
  @inbounds for idx=3:N
    n=idx-2
    Hn[idx]=x*sqrt(2/(n+1))*Hn[idx-1]- sqrt(n/(n+1))*Hn[idx-2]
  end
  return Hn*exp(x^2/2)
end



 # Hermite functions
 function HermiteHdx{T}(x::T,N::Int)
   Hdxn=Array(T, N)
   H0::T=1.0
   H1::T=sqrt(2)*x
   Hdxn[1]=-x   #-1./sqrt(2)*H1
   @inbounds for idx=2:N
     n=idx-1
     H2::T=x*sqrt(2/(n+1))*H1- sqrt(n/(n+1))*H0
     Hdxn[idx]=sqrt(n/2)*H0 -sqrt((n+1)/2)*H2
     H0=H1; H1=H2
   end
   return Hdxn.*exp(-x^2/2)
 end

function HermiteHdx_series{T}( x::Array{T,1}, u::Array{T,1} )
  E=Array(T,length(x))
  Np=length(x)
  Nx=length(u)
  @inbounds for pdx=1:Np
      E[pdx]=dot(HermiteHdx(x[pdx],Nx),u)
    end
  return E
end



function diff_Hermite{T}(H::Array{T,1})
  N=length(H)-1
  dH=zeros(T,length(H))
  dH[1]=1/sqrt(2)*H[2]
  dH[end]=-sqrt(N/2)*H[end-1]
  for idx=2:N-1
    n=idx-1
    dH[idx]=sqrt((n+1)/2)*H[idx+1] -sqrt(n/2)*H[idx-1]
  end
  return dH
end


function poissonK_Hermite(N::Int)
  K=spdiagm([-sqrt( (3:N).*((3:N)-1)*pi)/2,
             ((1:N)+0.5)*sqrt(pi)  ,
             -sqrt(((1:N-2)+2).*((1:N-2)+1)*pi)/2],
                   [2,0,-2])
  return K
end


@inbounds function  rhs_particle_Hermite{T}(xk::Array{T,1},wk::Array{T,1}, Nx::Int)
    const Np=length(xk)
    rho=zeros(T,Nx)
    # Precalculate coefficients
    alpha=Array(T,Nx-2)
    beta=Array(T,Nx-2)
    @inbounds for n=1:Nx-2
      alpha[n]=sqrt(2/(n+1))
      beta[n]=-sqrt(n/(n+1))
    end

    @inbounds for pdx=1:Np
      w::T=exp(-xk[pdx].^2/2)*wk[pdx]
      P0::T=1.0
      P1::T=sqrt(2)*P0*xk[pdx]
      rho[1]+=P0.*w
      rho[2]+=P1.*w
      for idx=1:Nx-2
        P2::T=xk[pdx].*alpha[idx].*P1+ beta[idx].*P0
        P0=P1; P1=P2;
        rho[idx+2]+=P2.*w
      end
    end
  return rho
end

@inbounds function HermiteH_series{T}(x::Array{T,1}, u::Array{T,1})
  N::Int=length(u)-1;
  alpha=Array(T,N+1)
  beta=Array(T,N+1)
  #
  for n=0:N
     alpha[n+1]=sqrt(2/(n+1))
     beta[n+1]=-sqrt((n+1)/(n+2))
  end
  E=Array(T,length(x))
  for pdx=1:length(x)
   b1::T=0
   b2::T=0
   for n=N:-1:0
     b0::T=u[n+1]+ x[pdx]*alpha[n+1]*b1 + beta[n+1]*b2
     b2=b1; b1=b0;
    end
    E[pdx]=b1*exp(-x[pdx]^2/2)
  end
  return E
end

# Coefficients for Runge Kutta
rksd=[2/3.0, -2/3.0, 1  ]
rksc=[ 7/24.0, 3/4.0, -1/24.0]


Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt)
kineticenergy=zeros(Float64,Nt)
momentum=zeros(Float64,Nt)
rho=zeros(Float64,Nx)
Phi=zeros(Float64,Nx+pad)
E=zeros(Float64,Nx+pad+2)
K=poissonK_Hermite(Nx+pad)

figure()
tic()
@inbounds for tdx=1:Nt
  if (mod(tdx-1,50)==0)
    PyPlot.clf()

    scatter(xk,vk,s=2, c=fk,lw=0)
    axis([-sigmaL*5,sigmaL*5,-3.5*vth,3.5*vth])

    sleep(0.01)
  end
  @inbounds for rkdx=1:length(rksd)

    # Bin to right hand side
    rhs=rhs_particle_Hermite( xk/sigmaL,wk,Nx)/Np
    # Hermite L^2 projection
    rho=rhs./sqrt(pi)
    # Remove ion background (quick'n dirty)
    rhs[1]=0
    Phi.=\(-K, [rhs;zeros(Float64,pad)])

    E.=diff_Hermite(gamma*[Phi;0;0])

    if rkdx==1
      fieldenergy[tdx]=(Phi'*K*Phi)[1]/2*sigmaL  #sum(E.^2)*sqrt(pi) #
      kineticenergy[tdx]=0.5.*sum(vk.^2.*wk )/Np
      momentum[tdx]=sum(vk.*wk)/Np
    end

    vk.+=dt*rksc[rkdx]*HermiteH_series(xk/sigmaL,E)
    #vk.+=dt*rksc[rkdx]*HermiteHdx_series(xk/sigmaL,-Phi)
    xk.+=dt*rksd[rkdx].*vk

    # Reflecting boundary
  end

  #print(tdx)
end
toc()
ttime=(0:Nt-1)*dt

prefix="plots/hermite_sheath"
#prefix="legendre"

#
# xg=collect(linspace(minimum(xk),maximum(xk)))
# figure()
# plot(xg, HermiteH_series(xg/sigmaL,Phi))

# HermiteHdx_series(xk/sigmaL,-Phi)
# HermiteH_series(xk/sigmaL,E)
#
# figure()
# plot(abs(rhs))

# plot(abs(Phi))
# plot(abs(E))
# plot(1:Nx,abs(rho))


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
# plot(kineticenergy)
# plot(kineticenergy+fieldenergy*sigmaL)



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
# savefig("$prefix\_fieldenergy.png")


# figure()
# semilogy(ttime,kineticenergy)
# grid();
# #autoscale(tight="True")
# ylabel("kinetic energy")
# xlabel("time")
# savefig("$prefix\_kineticenergy.png")

figure()
energy=kineticenergy+fieldenergy
semilogy(ttime,abs((energy-energy[1])./energy[1]))
grid()
ylabel("rel. energy error")
xlabel("time ")
# savefig("$prefix\_energy_error.png")
figure()
semilogy(ttime,abs(momentum-momentum[1]))
xlabel("time "); grid()
ylabel("absolute momentum error")
# savefig("$prefix\_momentum_error.png")


figure()
scatter(xk,vk,s=1, c=fk,lw=0)
# autoscale(tight="True")
# axis([0,L,minimum(vk),maximum(vk)])
colorbar();tight_layout()
xlabel("x");ylabel("v")
# savefig("$prefix\_particles.png")




#
# #Test OSDE HERMITE
# h=x->exp(-x.^2/2).*(x-0.5)+exp(-(x-2.2).^2)
# Dh=x->exp(-x.^2/2).*(x-0.5).*(-x) + exp(-x.^2/2)  -exp(-(x-2.2).^2).*2.*(x-2.2)
# Np=Int(1e7)
# xk=randn(Float64,Np)
#
# wk=h(xk)./(exp(-0.5.*xk.^2)/sqrt(2*pi) )
#
#
#
# hhat=rhs_particle_Hermite(xk,wk,40)/Np/sqrt(pi)
#
# x=collect(linspace(-15,15,2000))
# figure()
# plot(x,h(x))
# plot(x,HermiteH_series(x,hhat))
#
# figure()
# # plot(x,Dh(x))
# plot(x,gradient(h(x),x[2]-x[1] ))
#
# Dhhat=diff_Hermite(hhat)
#
#
# plot(x,HermiteH_series(x,Dhhat))
#  plot(x,HermiteHdx_series(x,hhat))
