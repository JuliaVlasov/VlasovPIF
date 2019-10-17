import MPI
using FastTransforms: cheb2leg, leg2cheb
using ApproxFun

#using Plots
Np=Int(1e4)
Nx=Int(32)   #Int(ceil(64*pi/2))
k0=0.5
dt=0.1
tmax=15
L=2*pi/k0

print("Asymptotically identical to ", floor(Nx/pi), " Fourier modes\n\n")
function f(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.1.*cos(k0*x))
end


MPI.Init()

comm = MPI.COMM_WORLD
MPI.Barrier(comm)
mpi_rank = MPI.Comm_rank(comm)
mpi_size = MPI.Comm_size(comm)
#print(mpi_size)

Np_loc=Int(floor(Np/mpi_size))

#Seed the random generator
#We would like to skip ahead values
srand( (Np_loc*4)*(mpi_rank+1) )

xk=rand(Np_loc)*L
vk=randn(Np_loc)
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


#@time LegendreP(xk[1],32)
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

function  rhs_particle2{T}(xk::Array{T,1},wk::Array{T,1}, Nx::Int)
    const Np=length(xk)
    rho=zeros(T,Nx)
    Pn=Array(T, Nx)
    Pn[1]=1.0

    @inbounds for pdx=1:Np
      Pn[2]=xk[pdx]
      @inbounds for idx=3:Nx
        n=idx-2
        Pn[idx]= Pn[2].*(2*n+1)/(n+1).*Pn[idx-1]- n/(n+1).*Pn[idx-2]
      end
      rho+=Pn.*wk[pdx]
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

#using BenchmarkTools
#t=@benchmark rhs=rhs_particle( xk/L*2-1,wk,Nx)/Np*(2/L)
#t2=@benchmark rhs=rhs_particle2( xk/L*2-1,wk,Nx)/Np*(2/L)
#t=@benchmark rhs2=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np*(2/L)
#@time rhs2=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np*(2/L)
#@time rhs=rhs_particle2( xk/L*2-1,wk,Nx)/Np*(2/L)



OM=Interval(0,L) # Domain
S=ApproxFun.Space(OM) #Default chebyshev space
B=periodic(S,0) # Periodic boundary
B2=DefiniteIntegral(S) # Nullspace condition
D2=-Derivative(S,2) #Laplace operator
D1=Derivative(S,1) # First derivative for H1 norm
IN=DefiniteIntegral(S) # First derivative for L2 norm
#Phi = \([B;B2;D2],[0.0;0.0;rho];tolerance=1E-16,maxlength=Nx+2)
QR = qrfact([B;B2;D2])

#OP=[B;B2;D2]
#sparse(IN[1:Nx,1:Nx])
#det(sparse(D2[1:Nx,1:Nx]+B2[1:Nx,1:Nx]+B[1:Nx,1:Nx]))
#AbstractMatrix(D2)

# Coefficients for Runge Kutta
const rksd=[2/3.0, -2/3.0, 1  ]
const rksc=[ 7/24.0, 3/4.0, -1/24.0]

Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt)
kineticenergy=zeros(Float64,Nt)
momentum=zeros(Float64,Nt)
rho=Fun(S)
Phi=Fun(S)
rhs=Array(Float64,Nx)
rhs=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np*(2/L)
rho=Fun(S,leg2cheb(rhs))
Phi=QR \[0.0;0.0;rho]
fieldenergy[1]=coefficients(IN*(D1*Phi)^2)[1]/2
Phic=-coefficients(Phi)[1:Nx+2]/L*2



#rho=Fun(S,leg2cheb(rand(Nx*64) ))
#using BenchmarkTools
#@benchmark Phi = \([B;B2;D2],[0.0;0.0;rho];tolerance=1E-16, maxlength=Nx+2)
#@benchmark Phi=QR \[0.0;0.0;rho]


MPI.Barrier(comm)
if (mpi_rank==0)
tic()
end
@inbounds for tdx=1:Nt

  @inbounds for rkdx=1:length(rksd)

    # Bin to right hand side
    rhs=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np*(2/L)
    rhs=MPI.allreduce(rhs, MPI.SUM, comm)

    #rhs=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np*(2/L)
    # Legendre L^2 projection
    rhs=rhs./(2./(2*(0:Nx-1)+1.0))
    # Remove ion background (quick'n dirty)
    rhs[1]=0
    #Transform to Chebyshev
    rho=Fun(S,leg2cheb(rhs))

    # Solve periodic Poisson equation with Chebyshev
    # Chebyshev periodic Poisson
    #Phi = \([B;B2;D2],[0.0;0.0;rho];tolerance=1E-16, maxlength=Nx+2)
    Phi=QR \[0.0;0.0;rho]
    #,maxlength=Nx+2

    #E=-rho./(-im*kx)
    if rkdx==1 & mpi_rank==0
      fieldenergy[tdx]=coefficients(IN*(D1*Phi)^2)[1]/2
      kineticenergy[tdx]=0.5.*sum(vk.^2.*wk )/Np
      momentum[tdx]=sum(vk.*wk)/Np
    end
    #Phic=-coefficients(Phi)[1:Nx+2]/L*2
    Phic=Phi.coefficients/L*2
    #@inbounds for pdx=1:Np_loc
    #vk[pdx]+=dt*rksd[rkdx]*chebTdx_series(xk[pdx]./(L*2.)-1.0, Phic)
    #vk+=dt*rksd[rkdx].*map(x->chebTdx_series(x, Phic), xk./(L*2.)-1.0)
  #  end
    vk.+=dt*rksd[rkdx]*chebTdx_series(xk/L*2-1.0,Phic )
    xk.+=dt*rksc[rkdx].*vk
    xk=mod(xk,L)
  end
end
MPI.Barrier(comm)

if (mpi_rank==0)
   toc()

#print("test: ", sum(fieldenergy))
using PyPlot
figure()
ttime=(0:Nt-1)*dt
semilogy(ttime,fieldenergy)
grid()
xlabel("time")
ylabel("electrostatic energy")
end
#MPI.Finalize()

# E=vk
# @benchmark chebTdx_series(xk/L*2-1.0,Phic )
#
# @benchmark rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np*(2/L)
