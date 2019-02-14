


#Some tools for Particle in Fourier
function eval_fourier_modes{T}(arg::T, N::Int)
    mode=Array(Complex{T},N)
    psin1=(cos(arg)+ im*sin(arg))
    psin=1.0+im*0.0
    for kdx=1:N
       psin=psin*psin1
       @inbounds mode[kdx]=psin
     end
     return mode
end

function eval_fourier_modes{T<:AbstractFloat}(arg::T, N::Int,
                           mode::Array{Complex{T},1})
    psin1=(cos(arg)+ im*sin(arg))
    psin=1.0+im*0.0
    @inbounds for kdx=1:N
       psin=psin*psin1
       @inbounds mode[kdx]=psin
     end
     nothing
end


function eval_fourier_modes0{T<:AbstractFloat}(arg::T, N::Int,
                           mode::Array{Complex{T},1})
    const psin1=(cos(arg)+ im*sin(arg))
    psin=1.0+im*0.0
    mode[1]=psin
    @inbounds for kdx=2:(N+1)
       psin=psin*psin1
       @inbounds mode[kdx]=psin
     end
     nothing
end

function eval_fourier_modes2{T<:AbstractFloat}(arg::T, N::Int)
    mode=Array(Complex{T},2*N+1)
    N0=N+1
    mode[N0]=1.0
    psin1=(cos(arg)+ im*sin(arg))
    psin=1.0+im*0.0
    @inbounds for kdx=1:N
       psin=psin*psin1
       @inbounds mode[N0+kdx]=psin
                 mode[N0-kdx]=conj(psin)
     end
    return mode
end
function eval_fourier_modes2{T<:AbstractFloat}(arg::T, N::Int,
                           mode::Array{Complex{T},1})
    N0::Int=N+1
    mode[N0]=1.0
    const psin1=(cos(arg)+ im*sin(arg))::Complex{T}
    psin=(1.0+im*0.0)::Complex{T}
    @inbounds for kdx=1:N
       psin=psin*psin1
       @inbounds mode[N0+kdx]=psin
                 mode[N0-kdx]=conj(psin)
     end
     nothing
end
# function eval_fourier_modes2{T<:AbstractFloat}(arg::T, N::Int,
#                            mode::Array{Complex{T},1})
#     N0=N+1
#     mode[N0]=1.0
#     mode[N0+1]=(cos(arg)- im*sin(arg))::Complex{T}
#     mode[N0-1]=conj(mode[N0+1])
#     @inbounds for kdx=2:N
#        mode[N0+kdx]=mode[N0+kdx-1]*mode[N0]
#        mode[N0-kdx]=conj(mode[N0-kdx])
#      end
#      nothing
# end


# General structure for a 3d PIF field solver
type pif3d
  Nx1::Int
  Nx2::Int
  Nx3::Int
  L1::Float64
  L2::Float64
  L3::Float64
  N1::Int
  N2::Int
  N3::Int
  function pif3d(Nx1,Nx2,Nx3,L1,L2,L3)
    new(Nx1,Nx2,Nx3,L1,L2,L3,Nx1+1,Nx2*2+1,Nx3*2+1)
  end
  function pif3d(Nx1::Int,Nx2::Int,Nx3::Int,L::Float64)
    pif3d(Nx1,Nx2,Nx3,L,L,L)
  end
end


#
# function rhs_particle0{T <:AbstractFloat}(self::pif3d,
#            x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},  wn::Array{T,1})
#   Np=length(x1n)
#   @assert Np==length(x2n)
#   @assert Np==length(x3n)
#   @assert Np==length(wn)
#
#
#   rhs=zeros(Complex{T},self.N1,self.N2,self.N3)
#
#   k1=2*pi/self.L1
#   k2=2*pi/self.L2
#   k3=2*pi/self.L3
#
#   @inbounds begin
#   for pdx=1:Np
#     #Julia is column major
#
#     for kdx=1:self.N3
#        for jdx=1:self.N2
#         @simd  for idx=1:self.N1
#           arg::Complex{T}=k1*x1n[pdx]+k2*x2n[pdx]+k3*x3n[pdx]
#            rhs[idx,jdx,kdx]+=(cos(arg)+ im*sin(arg))*wn[pdx]
#         end
#       end
#     end
#
#   end
#   end
#   return conj(rhs)/self.L1/self.L2/self.L3
# end

function rhs_particle{T <:AbstractFloat}(self::pif3d,
           x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},  wn::Array{T,1})
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)
  @assert Np==length(wn)


  rhs=zeros(Complex{T},self.N1,self.N2,self.N3)

  k1=2*pi/self.L1
  k2=2*pi/self.L2
  k3=2*pi/self.L3

  @inbounds begin
  for pdx=1:Np
    #Julia is column major
    eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)

    #Introduce weight
    @simd for idx=1:self.N1
    psi1[idx]*=wn[pdx]
    end

    for kdx=1:self.N3
       for jdx=1:self.N2
        psi23::Complex{T}=psi2[jdx]*psi3[kdx]
        @simd  for idx=1:self.N1
           rhs[idx,jdx,kdx]+=psi1[idx]*psi23
        end
      end
    end
    #Accumulate
    # for idx=1:self.N1
    #   for jdx=1:self.N2
    #     @simd for kdx=1:self.N3
    #       rhs[idx,jdx,kdx]+=psi1[idx]*psi2[jdx]*psi3[kdx]
    #     end
    #   end
    # end
  end
  end
  return conj(rhs)/self.L1/self.L2/self.L3
end



function rhs_particle2{T <:AbstractFloat}(self::pif3d,
           x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},  wn::Array{T,1})
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)
  @assert Np==length(wn)


  rhs=zeros(Complex{T},self.N1,self.N2,self.N3)
  psi1=Array(Complex{T},self.N1)
  psi2=Array(Complex{T},self.N2)
  psi3=Array(Complex{T},self.N3)

  k1=2*pi/self.L1
  k2=2*pi/self.L2
  k3=2*pi/self.L3

  @inbounds begin
  for pdx=1:Np
    #Julia is column major
    eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)

    #Introduce weight
    psi1*=wn[pdx]
    for kdx=1:self.N3
       BLAS.ger!(psi3[kdx],psi1,psi2,rhs[:,:,kdx])
    end
    #for idx=1:self.N1
    #  BLAS.ger!(psi1[idx],psi2,psi3,rhs[idx,:,:])
    #end

  end
  end
  return conj(rhs)/self.L1/self.L2/self.L3
end


function eval_scalar{T}(self::pif3d,
           x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
           F::Array{Complex{T},3})
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)

  Fn=zeros(Complex{T},Np)
  psi1=Array(Complex{T},self.N1)
  psi2=Array(Complex{T},self.N2)
  psi3=Array(Complex{T},self.N3)
  k1=2*pi/self.L1
  k2=2*pi/self.L2
  k3=2*pi/self.L3

  for pdx=1:Np
    eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)
    for kdx=1:self.N3
       for jdx=1:self.N2
        psi23::Complex{T}=psi2[jdx]*psi3[kdx]
        Fn[pdx]+=psi23*F[1,jdx,kdx]
        @simd  for idx=2:self.N1
          psin::Complex{T}=psi1[idx]*psi23
           Fn[pdx]+=(real(psin)*real(F[idx,jdx,kdx])
                    -imag(psin)*imag(F[idx,jdx,kdx]))/2.0
        end
      end
    end

  end
  return real(Fn)
end


function eval_scalar{T}(self::pif3d,
           x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
           F::Array{Array{Complex{T},3}})
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)

  D=length(F)
  Fn=zeros(Complex{T},Np,D)
  psi1=Array(Complex{T},self.N1)
  psi2=Array(Complex{T},self.N2)
  psi3=Array(Complex{T},self.N3)
  k1=2*pi/self.L1
  k2=2*pi/self.L2
  k3=2*pi/self.L3

  for pdx=1:Np
    eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)
    for kdx=1:self.N3
       for jdx=1:self.N2
        psi23::Complex{T}=psi2[jdx]*psi3[kdx]
        for ddx=1:D
            Fn[pdx,ddx]+=psi23*F[ddx][1,jdx,kdx]
            end
            for ddx=1:D
          @simd  for idx=2:self.N1
            psin::Complex{T}=psi1[idx]*psi23
            Fn[pdx,ddx]+=(real(psin)*real(F[ddx][idx,jdx,kdx])
                          -imag(psin)*imag(F[ddx][idx,jdx,kdx]))/2.0
          end
        end
      end
    end

  end
  return real(Fn)
end

function getK(self::pif3d)
  K1=collect(0:self.Nx1)*self.L1*2*pi
  K2=collect(-self.Nx2:self.Nx2)*self.L2*2*pi
  K3=collect(-self.Nx3:self.Nx3)*self.L3*2*pi
  return K1,K2,K3
end

function getK2(self::pif3d)
  K1,K2,K3=getK(self)
  K1=reshape(K1,self.N1,1,1)
  K2=reshape(K2,1,self.N2,1)
  K3=reshape(K3,1,1,self.N3)
  return K1,K2,K3
end



function solve_poisson!{T}(self::pif3d,
                      rho::Array{Complex{T},3}, Phi::Array{Complex{T},3})
    K1,K2,K3=getK(self)
     for kdx=1:self.N3
        for jdx=1:self.N2
          for idx=1:self.N1
           if (idx!=1 && jdx!=self.Nx2+1  && kdx!=self.Nx3+1 )

            Phi[idx,jdx,kdx]=1/(K1[idx]^2 +K2[jdx]^2 +K3[kdx]^2)
          end
         end
       end
     end
end
function solve_poisson{T}(self::pif3d, rho::Array{Complex{T},3})
  Phi=zeros(rho)
  solve_poisson!(self,rho,Phi)
  return Phi
end

function solve_gauss!{T}(self::pif3d, rho::Array{Complex{T},3}, E1,E2,E3   )
  Phi=solve_poisson(self,rho)
  K1,K2,K3=getK2(self)
  E1=-Phi*im.*K1
  E2=-Phi*im.*K2
  E3=-Phi*im.*K3
end

# Gradient of a scalar function
function gradient{T}(self::pif3d, F::Array{Complex{T},3})
  K1,K2,K3=getK2(self)
  DF1=F*im.*K2
  DF2=F*im.*K2
  DF3=F*im.*K3
  return DF1,DF2,DF3
end

function div{T}(self::pif3d, F::Array{Complex{T},3})
  DF1,DF2,DF3=gradient(pif3d, F)
  return DF1.+DF2.+DF3
end


function curl{T}(self::pif3d, F1::Array{Complex{T},3},
              F2::Array{Complex{T},3},F3::Array{Complex{T},3} )
    K1,K2,K3=getK2(self)
    return im*(K2.*F3 - K3.*F2), im*(K3.*F1 - K1.*F3), im*(K1.*F2 - K2.*F1)
end










Np=Int(1e5)
x1n=rand(Float64,Np)
x2n=rand(Float64,Np)
x3n=rand(Float64,Np)

# psix1=Array(Complex128,Nx1*2+1)
# @time eval_fourier_modes2(x1n[1]*2*pi,Nx1,psix1)
# @time psix1=eval_fourier_modes2(x1n[1]*2*pi,Nx1)

# psix1*reshape(psix1,1,Nf1)
wn=x1n

self=pif3d(10,20,1, 1.0)

# wn=f0(x1n,x2n,x3n,vn)./g0(x1n,x2n,x3n,vn)
@time rho=rhs_particle(self,x1n,x2n,x3n,wn)/Np
@time rho=rhs_particle2(self,x1n,x2n,x3n,wn)/Np

sum(abs(rho-rho2)[:])
Phi=zeros(rho)
Phi=solve_poisson(self, rho)


E1,E2,E3=gradient(self,-Phi)


E=[E1,E2,E3]

E[1,2][1]
@time (E1n=eval_scalar(self,x1n,x2n,x3n,E2);E2n=eval_scalar(self,x1n,x2n,x3n,E2);E3n=eval_scalar(self,x1n,x2n,x3n,E3))

@time En=eval_scalar(self,x1n,x2n,x3n,[E1,E2,E3])

k0=0.5
dt=0.1
tmax=15

L=2*pi/k0


function f0(xi1,xi2,xi3,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.1.*cos(xi1))
end

function g0(xi1,xi2,xi3,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi)
end
