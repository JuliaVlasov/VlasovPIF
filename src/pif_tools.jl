#Some tools for Particle in Fourier in three dimensions

export pm_pif3d

#Diagnostics
export L2norm, H1norm, dot

# Vlasov--Poisson
export accum_osde, accum_osde_var
export solve_poisson!,solve_poisson, gradient
export integrate_H_E
export integrate_vxB

# Vlasov--Maxwell
export integrate_H_B,integrate_Faraday
export integrate_Hp_split_sym, integrate_Hp, integrate_Hp_boris_exp
export gauss_error

export to_grid

function eval_fourier_modes(arg::T, N::Int) where {T}
    mode=Array{Complex{T}}(N)
    psin1=(cos(arg)+ im*sin(arg))
    psin=one(Complex{T})
    for kdx=1:N
       psin=psin*psin1
       @inbounds mode[kdx]=psin
     end
     return mode
end

function eval_fourier_modes(arg::T, N::Int, mode::Array{Complex{T},1}) where {T}
    psin1::Complex{T}=cis(arg)
    psin::Complex{T}=one(Complex{T})
    @inbounds for kdx=1:N
       psin=psin*psin1
       @inbounds mode[kdx]=psin
     end
     nothing
end

function eval_fourier_modes0(arg::T, N::Int ) where {T <: AbstractFloat}
    mode=Array{Complex{T}}(N+1)
    eval_fourier_modes0(arg,N,mode)
    return mode
end

function eval_fourier_modes0(arg::T, N::Int,
                           mode::Array{Complex{T},1}) where {T <: AbstractFloat}
    psin1::Complex{T}=cis(arg)
    psin::Complex{T}=one(Complex{T})
    mode[1]=psin
    @inbounds for kdx=2:(N+1)
       psin=psin*psin1
       @inbounds mode[kdx]=psin
     end
     nothing
end

function eval_fourier_modes2(arg::T, N::Int) where {T <: AbstractFloat}
    mode=Array{Complex{T}}(2*N+1)
    N0=N+1
    psin1 :: Complex{T} = cis(arg)
    psin  :: Complex{T} = one(Complex{T})
    mode[N0]=one(Complex{T})
    @inbounds for kdx=1:N
       psin=psin*psin1
       @inbounds mode[N0+kdx]=psin
                 mode[N0-kdx]=conj(psin)
     end
    return mode
end

function eval_fourier_modes2(arg::T, N::Int,
                           mode::Array{Complex{T},1}) where {T <: AbstractFloat}
    N0::Int=N+1
    mode[N0]=one(Complex{T})
    psin1 :: Complex{T} = cis(arg)
    psin  :: Complex{T} = one(Complex{T})
    @inbounds for kdx=1:N
       psin=psin*psin1
       @inbounds mode[N0+kdx]=psin
                 mode[N0-kdx]=conj(psin)
     end
     nothing
end
function sumcnj(a::Complex{T},b::Complex{T}) where {T}
  c::T= (real(a)*real(b)-imag(a)*imag(b))*2
  return c
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
mutable struct pm_pif3d{T}
  Nx1::Int
  Nx2::Int
  Nx3::Int
  L1::T
  L2::T
  L3::T
  N1::Int
  N2::Int
  N3::Int
  # Unit mode
  kx1::T
  kx2::T
  kx3::T
  K1::Array{T}{1}
  K2::Array{T}{1}
  K3::Array{T}{1}
  # Indicies of zero modes
  N1n::Int
  N2n::Int
  N3n::Int

  function pm_pif3d(Nx1::Int,Nx2::Int,Nx3::Int,
                                L1::T,L2::T,L3::T) where {T<:AbstractFloat}


    kx1 = 2*pi / self.L1
    kx2 = 2*pi / self.L2
    kx3 = 2*pi / self.L3
    N1n = 1
    N2n = Nx2+1
    N3n = Nx3+1

    K1 = collect(0:Nx1)    ./ L1 * 2 * pi
    K2 = collect(-Nx2:Nx2) ./ L2 * 2 * pi
    K3 = collect(-Nx3:Nx3) ./ L3 * 2 * pi

    new(Nx1,Nx2,Nx3,L1,L2,L3,Nx1+1,Nx2*2+1,Nx3*2+1, kx1, kx2, kx3, K1, K2, K3, N1n, N2n, N3n)

  end


end

function pm_pif3d(Nx1::Int,Nx2::Int,Nx3::Int,L::T) where {T}
    pm_pif3d{T}(Nx1,Nx2,Nx3,L,L,L)
end

# Outer constructor to infer type T directly from given length
pm_pif3d(Nx1::Int,Nx2::Int,Nx3::Int,L1::T,L2::T,L3::T) where
      {T}=pm_pif3d{T}(Nx1,Nx2,Nx3,L1::T,L2::T,L3::T)
pm_pif3d(Nx1::Int,Nx2::Int,Nx3::Int,L::T) where
            {T<:AbstractFloat}=pm_pif3d{T}(Nx1,Nx2,Nx3,L)


function eval_basis2(self::pm_pif3d, x1n::T,x2n::T,  x3n::T,
                            psin::Array{Complex{T},3}, 
                            psi1::Array{Complex{T},1},
                            psi2::Array{Complex{T},1},
                            psi3::Array{Complex{T},1}) where {T}

  eval_fourier_modes0(x1n*self.kx1, self.Nx1,psi1)
  eval_fourier_modes2(x2n*self.kx2, self.Nx2,psi2)
  eval_fourier_modes2(x3n*self.kx3, self.Nx3,psi3)

  #psin.=(reshape(psi1,self.N1,1,1)*
  #     reshape(psi2,1,self.N2,1)*reshape(psi3,1,1,self.N3))
       for kdx=1:self.N3
          for jdx=1:self.N2
           psi23::Complex{T}=psi2[jdx]*psi3[kdx]
           @simd  for idx=1:self.N1
              psin[idx,jdx,kdx]=psi1[idx]*psi23
           end
         end
       end

end

@inline @inbounds function eval_basis(self::pm_pif3d, x1n::T,x2n::T,  x3n::T,
                            psin::Array{Complex{T},3}) where {T}


  mode1::Complex{T}=cis(x1n*self.kx1)
  mode2::Complex{T}=cis(x2n*self.kx2)
  mode3::Complex{T}=cis(x3n*self.kx3)
  mode2c::Complex{T}=conj(mode2)
  mode3c::Complex{T}=conj(mode3)

  # k3=0,k2=0,k1=0
  psin[1,self.N2n,self.N3n]=one(Complex{T})
  # k3=0,k2=0, k1!=0
  for idx=1:self.Nx1
     psin[idx+1,self.N2n,self.N3n]=psin[idx,self.N2n,self.N3n]*mode1
  end
  for jdx=1:self.Nx2
    psin[1,self.N2n+jdx,self.N3n]=psin[1,self.N2n+(jdx-1),self.N3n]*mode2
    for idx=1:self.Nx1
       psin[1+idx,self.N2n+jdx,self.N3n]=psin[idx,self.N2n+jdx,self.N3n]*mode1
    end
    psin[1,self.N2n-jdx,self.N3n]=psin[1,self.N2n-(jdx-1),self.N3n]*mode2c
    for idx=1:self.Nx1
       psin[1+idx,self.N2n-jdx,self.N3n]=psin[idx,self.N2n-jdx,self.N3n]*mode1
    end
  end

  for kdx=1:self.Nx3

    psin[1,self.N2n,self.N3n+kdx]=psin[1,self.N2n,self.N3n+(kdx-1) ]*mode3
    for idx=1:self.Nx1
       psin[idx+1,self.N2n,self.N3n+kdx]=psin[idx,self.N2n,self.N3n+kdx]*mode1
    end
    for jdx=1:self.Nx2
      psin[1,self.N2n+jdx,self.N3n+kdx]=psin[1,self.N2n+(jdx-1),self.N3n+kdx]*mode2
      for idx=1:self.Nx1
         psin[idx+1,self.N2n+jdx,self.N3n+kdx]=psin[idx,self.N2n+jdx,self.N3n+kdx]*mode1
      end
      psin[1,self.N2n-jdx,self.N3n+kdx]=psin[1,self.N2n-(jdx-1),self.N3n+kdx]*mode2c
      for idx=1:self.Nx1
         psin[idx+1,self.N2n-jdx,self.N3n+kdx]=psin[idx,self.N2n-jdx,self.N3n+kdx]*mode1
      end
    end

    psin[1,self.N2n,self.N3n-kdx]=psin[1,self.N2n,self.N3n-(kdx-1) ]*mode3c
    for idx=1:self.Nx1
       psin[idx+1,self.N2n,self.N3n-kdx]=psin[idx,self.N2n,self.N3n-kdx]*mode1
    end
    for jdx=1:self.Nx2
      psin[1,self.N2n+jdx,self.N3n-kdx]=psin[1,self.N2n+(jdx-1),self.N3n-kdx]*mode2
      for idx=1:self.Nx1
         psin[idx+1,self.N2n+jdx,self.N3n-kdx]=psin[idx,self.N2n+jdx,self.N3n-kdx]*mode1
      end
      psin[1,self.N2n-jdx,self.N3n-kdx]=psin[1,self.N2n-(jdx-1),self.N3n-kdx]*mode2c
      for idx=1:self.Nx1
         psin[idx+1,self.N2n-jdx,self.N3n-kdx]=psin[idx,self.N2n-jdx,self.N3n-kdx]*mode1
      end
    end
  end
end
@inline @inbounds function eval_basis(self::pm_pif3d{T}, x1n::T,x2n::T,  x3n::T,
                            psin::Array{Complex{T},3}) where {T}

  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)

  eval_fourier_modes0(x1n*self.kx1, self.Nx1,psi1)
  eval_fourier_modes2(x2n*self.kx2, self.Nx2,psi2)
  eval_fourier_modes2(x3n*self.kx3, self.Nx3,psi3)

  #psin.=(reshape(psi1,self.N1,1,1)*
  #     reshape(psi2,1,self.N2,1)*reshape(psi3,1,1,self.N3))
       for kdx=1:self.N3
          for jdx=1:self.N2
           psi23::Complex{T}=psi2[jdx]*psi3[kdx]
            for idx=1:self.N1
              psin[idx,jdx,kdx]=psi1[idx]*psi23
           end
         end
       end

end


# Definite Integral from xIn to x_n on axis I
# The other components remain constant
function eval_basis_defint{T}(self::pm_pif3d{T}, x1n::T,x2n::T, x3n::T,x_n::T,
                          axis::Int,psin::Array{Complex{T},3})


  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)

  eval_fourier_modes0(x1n*2*pi/self.L1, self.Nx1,psi1)
  eval_fourier_modes2(x2n*2*pi/self.L2, self.Nx2,psi2)
  eval_fourier_modes2(x3n*2*pi/self.L3, self.Nx3,psi3)


  K=getK(self,axis)
  if (axis==1)
    K[1]=1.0
    psi1=(eval_fourier_modes0(x_n*2*pi/self.L1, self.Nx1)-psi1)
    psi1./=(im*K)
    psi1[1]=x_n-x1n
  elseif (axis==2)
    K[self.Nx2+1]=1.0
    psi2=(eval_fourier_modes2(x_n*2*pi/self.L1, self.Nx2)-psi2)
    psi2./=(im*K)
    psi2[self.Nx2+1]=x_n-x2n
  elseif (axis==3)
    K[self.Nx3+1]=1.0
    psi3=(eval_fourier_modes2(x_n*2*pi/self.L1, self.Nx3)-psi3)
    psi3./=(im*K)
    psi3[self.Nx3+1]=x_n-x3n
  end

  #psin.=(reshape(psi1,self.N1,1,1)*
  #     reshape(psi2,1,self.N2,1)*reshape(psi3,1,1,self.N3))
       for kdx=1:self.N3
          for jdx=1:self.N2
           psi23::Complex{T}=psi2[jdx]*psi3[kdx]
           @simd  for idx=1:self.N1
              psin[idx,jdx,kdx]=psi1[idx]*psi23
           end
         end
       end

end

# Indefinite Integral from xIn to x_n on axis I
# The other components remain constant
function eval_basis_int{T}(self::pm_pif3d{T}, x1n::T,x2n::T, x3n::T,
                          axis::Int,psin::Array{Complex{T},3})


  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)

  eval_fourier_modes0(x1n*2*pi/self.L1, self.Nx1,psi1)
  eval_fourier_modes2(x2n*2*pi/self.L2, self.Nx2,psi2)
  eval_fourier_modes2(x3n*2*pi/self.L3, self.Nx3,psi3)


  K=getK(self,axis)
  if (axis==1)
    K[1]=1.0
    psi1=(eval_fourier_modes0(x_n*2*pi/self.L1, self.Nx1)-psi1)
    psi1./=(im*K)
    psi1[1]=x1n
  elseif (axis==2)
    K[self.Nx2+1]=1.0
    psi2=(eval_fourier_modes2(x_n*2*pi/self.L1, self.Nx2)-psi2)
    psi2./=(im*K)
    psi2[self.Nx2+1]=x2n
  elseif (axis==3)
    K[self.Nx3+1]=1.0
    psi3=(eval_fourier_modes2(x_n*2*pi/self.L1, self.Nx3)-psi3)
    psi3./=(im*K)
    psi3[self.Nx3+1]=x3n
  end

  #psin.=(reshape(psi1,self.N1,1,1)*
  #     reshape(psi2,1,self.N2,1)*reshape(psi3,1,1,self.N3))
       for kdx=1:self.N3
          for jdx=1:self.N2
           psi23::Complex{T}=psi2[jdx]*psi3[kdx]
           @simd  for idx=1:self.N1
              psin[idx,jdx,kdx]=psi1[idx]*psi23
           end
         end
       end

end

# evaluate a scalar function with coefficients F
@inbounds function eval_scalar{T}(self::pm_pif3d, psin::Array{Complex{T},3},
                   F::Array{Complex{T},3} )
  # vali::Complex{T}=0
  # val::T=0
  # for kdx=1:self.N3
  #    for jdx=1:self.N2
  #       vali+=psin[1,jdx,kdx]*F[1,jdx,kdx]
  #     @simd  for idx=2:self.N1
  #        val+=2*real(psin[idx,jdx,kdx])*real(F[idx,jdx,kdx])
  #        val-=2*imag(psin[idx,jdx,kdx])*imag(F[idx,jdx,kdx])
  #     end
  #   end
  # end
  # val+=real(vali)
  # return val
  val::Complex{T}=0
  for kdx=1:self.N3
     for jdx=1:self.N2
        val+=psin[1,jdx,kdx]*F[1,jdx,kdx]
      @simd  for idx=2:self.N1
         val+=2*real(psin[idx,jdx,kdx])*real(F[idx,jdx,kdx])
         val-=2*imag(psin[idx,jdx,kdx])*imag(F[idx,jdx,kdx])
      end
    end
  end
  valr::T=real(val)
  return valr
end
# avoiding complex numbers yields a bad memory access pattern
@inbounds function eval_scalar2{T}(self::pm_pif3d, psin::Array{Complex{T},3},
                     F::Array{Complex{T},3} )
  val::T=real(psin[1,self.N2n,self.N3n]*F[1,self.N2n,self.N3n])
  for jdx=1:self.Nx2
      val+=real( psin[1,self.N2n+jdx,self.N3n]*F[1,self.N2n+jdx, self.N3n]
                 +  psin[1,self.N2n-jdx,self.N3n]*F[1,self.N2n-jdx,self.N3n])
   @simd  for idx=2:self.N1
      val+=2*real(psin[idx,self.N2n+jdx,self.N3n])*real(F[idx,self.N2n+jdx,self.N3n])
      val-=2*imag(psin[idx,self.N2n+jdx,self.N3n])*imag(F[idx,self.N2n+jdx,self.N3n])
      val+=2*real(psin[idx,self.N2n-jdx,self.N3n])*real(F[idx,self.N2n-jdx,self.N3n])
      val-=2*imag(psin[idx,self.N2n-jdx,self.N3n])*imag(F[idx,self.N2n-jdx,self.N3n])
   end
  end


  for kdx=1:self.Nx3
    val+=real( psin[1,self.N2n,self.N3n+kdx]*F[1,self.N2n,self.N3n+kdx]
              +psin[1,self.N2n,self.N3n-kdx]*F[1,self.N2n,self.N3n-kdx])
    @simd  for idx=2:self.N1
       val+=2*real(psin[idx,self.N2n,self.N3n+kdx])*real(F[idx,self.N2n,self.N3n+kdx])
       val-=2*imag(psin[idx,self.N2n,self.N3n+kdx])*imag(F[idx,self.N2n,self.N3n+kdx])
    end
    @simd  for idx=2:self.N1
    val+=2*real(psin[idx,self.N2n,self.N3n-kdx])*real(F[idx,self.N2n,self.N3n-kdx])
    val-=2*imag(psin[idx,self.N2n,self.N3n-kdx])*imag(F[idx,self.N2n,self.N3n-kdx])
    end
    for jdx=1:self.Nx2
        val+=real( psin[1,self.N2n+jdx,self.N3n+kdx]*F[1,self.N2n+jdx, self.N3n+kdx]
                  + psin[1,self.N2n-jdx,self.N3n-kdx]*F[1,self.N2n-jdx,self.N3n-kdx])
        val+= real( psin[1,self.N2n+jdx,self.N3n-kdx]*F[1,self.N2n+jdx,self.N3n-kdx]
                  + psin[1,self.N2n-jdx,self.N3n+kdx]*F[1,self.N2n-jdx,self.N3n+kdx])
    @simd  for idx=2:self.N1
        val+=2*real(psin[idx,self.N2n+jdx,self.N3n+kdx])*real(F[idx,self.N2n+jdx,self.N3n+kdx])
        val-=2*imag(psin[idx,self.N2n+jdx,self.N3n+kdx])*imag(F[idx,self.N2n+jdx,self.N3n+kdx])
        val+=2*real(psin[idx,self.N2n-jdx,self.N3n+kdx])*real(F[idx,self.N2n-jdx,self.N3n+kdx])
        val-=2*imag(psin[idx,self.N2n-jdx,self.N3n+kdx])*imag(F[idx,self.N2n-jdx,self.N3n+kdx])
      end
    @simd  for idx=2:self.N1
        val+=2*real(psin[idx,self.N2n+jdx,self.N3n-kdx])*real(F[idx,self.N2n+jdx,self.N3n-kdx])
        val-=2*imag(psin[idx,self.N2n+jdx,self.N3n-kdx])*imag(F[idx,self.N2n+jdx,self.N3n-kdx])
        val+=2*real(psin[idx,self.N2n-jdx,self.N3n-kdx])*real(F[idx,self.N2n-jdx,self.N3n-kdx])
        val-=2*imag(psin[idx,self.N2n-jdx,self.N3n-kdx])*imag(F[idx,self.N2n-jdx,self.N3n-kdx])
     end
    end
  end
  return val

  # return real(sum(psin[1,:,:].*F[1,:,:])) +
  #          2*( dot(real(psin[2:end,:,:]),real(F[2:end,:,:])) +
  #           dot(imag(psin[2:end,:,:]),imag(F[2:end,:,:]) ) )
end


# evaluate a scalar function with coefficients F
@inbounds function eval_scalar_gradient{T}(self::pm_pif3d,
          psin::Array{Complex{T},3},
                        F::Array{Complex{T},3},gradF::Array{Complex{T},1})
      gradF[:]=0
       for kdx=1:self.N3
          dx3::T=( kdx-self.N3n)*pm.kx3
          for jdx=1:self.N2
            dx2::T=( jdx-self.N2n)*pm.kx2
            #gradF[1]+=0
            gradF[2]+=psin[1,jdx,kdx]*F[1,jdx,kdx]*im*dx2
            gradF[3]+=psin[1,jdx,kdx]*F[1,jdx,kdx]*im*dx3
           @simd  for idx=2:self.N1
             val::T=2*(imag(psin[idx,jdx,kdx])*imag(F[idx,jdx,kdx])
                       +real(psin[idx,jdx,kdx])*real(F[idx,jdx,kdx]))
             gradF[1]+=val*(idx-1)*pm.kx1
             gradF[2]+=val*dx2
             gradF[3]+=val*dx3
           end
         end
       end

  #gradF[:].im=0.0
end
function eval_scalar_gradient{T}(self::pm_pif3d,
          psin::Array{Complex{T},3},
                        F::Array{Complex{T},3})
  gradF=Array{Complex{T}}(3)
  eval_scalar_gradient(self,psin,F,gradF)
  return real(gradF)
end



# function set_basis_int{T}(self::pm_pif3d, psi::Array{Complex{T},3}, x::T,  axis::Int )
#      psi_int=copy(psi)
#      K=getK(self,axis)
#
#      if (axis==1)
#     psi_int[1,:,:]=
#
#      return psi_int
# end
# using Devectorize
function integrate_Hp{T}(self::pm_pif3d{T},x1n::Array{T,1},x2n::Array{T,1},
                       x3n::Array{T,1},
                       v1n::Array{T,1},v2n::Array{T,1}, v3n::Array{T,1},
                      qn::Array{T,1}, mn::Array{T,1},wn::Array{T,1},
          J1::Array{Complex{T},3},J2::Array{Complex{T},3},J3::Array{Complex{T},3},
          B1::Array{Complex{T},3},B2::Array{Complex{T},3}, B3::Array{Complex{T},3}, dt::T)
  Np::Int=length(x1n)

  psi1=Array{Complex{T}}(self.N1,self.N2,self.N3)
  psi0=Array{Complex{T}}(self.N1,self.N2,self.N3)
  # =Array{Complex{T}}(self.N1,self.N2,self.N3)

  psix1=Array{Complex{T}}(self.N1)
  psix2=Array{Complex{T}}(self.N2)
  psix3=Array{Complex{T}}(self.N3)

  #Fourier stem function Integral
  DK1=Array{Complex{T}}(self.N1)
  DK1[:]=getK(self,1)
  DK1 = 1. / (im*DK1)
  DK1[self.N1n]=0.0
  DK2=Array{Complex{T}}(self.N2)
  DK2[:]=getK(self,2)
  DK2 = 1. / (im*DK2)
  DK2[self.N2n]=0.0
  DK3=Array{Complex{T}}(self.N3)
  DK3[:]=getK(self,3)
  DK3 = 1. / (im*DK3)
  DK3[self.N3n]=0.0

  vol::T = 1. / (self.L1 * self.L2 * self.L3)

  for pdx=1:Np
    w::T=wn[pdx]*qn[pdx]*vol #Accumulate<d weight with fourier normalization
    q_m::T=qn[pdx]/mn[pdx]
    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi0)
    #Hp_1
    x1n[pdx]+=dt*v1n[pdx]
    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)
    #eval_fourier_modes0(dt*v1n[pdx]*pm.kx1, self.Nx1,psix1)
    # for kdx=1:self.N3
    #   for jdx=1:self.N2
    #     @simd for idx=1:self.N1
    #      psi1[idx,jdx,kdx]=psi0[idx,jdx,kdx]*psix1[idx]
    #     end
    #   end
    # end

    #Construct stem function
    for kdx=1:self.N3
      for jdx=1:self.N2
        psi0[1,jdx,kdx]=(dt*v1n[pdx]).*psi0[1,jdx,kdx]
        @simd for idx=2:self.N1
         psi0[idx,jdx,kdx]=(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx]).*DK1[idx]
        end
      end
    end

    v2n[pdx]+= (-q_m)*eval_scalar(self, psi0 ,B3)
    v3n[pdx]+= q_m*eval_scalar(self, psi0,B2)

    #Accumulation
    conj!(psi0)
    for kdx=1:self.N3
      for jdx=1:self.N2
       @simd for idx=1:self.N1
       J1[idx,jdx,kdx]+=psi0[idx,jdx,kdx]*w
       psi0[idx,jdx,kdx]=psi1[idx,jdx,kdx]
       end
      end
    end

    #Hp_2
    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi0)
    x2n[pdx]+=dt*v2n[pdx]
    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)
    # eval_fourier_modes2(dt*v2n[pdx]*pm.kx2, self.Nx2,psix2)
    # for kdx=1:self.N3
    #   for jdx=1:self.N2
    #     @simd for idx=1:self.N1
    #      psi1[idx,jdx,kdx]=psi0[idx,jdx,kdx]*psix2[jdx]
    #     end
    #   end
    # end

    #Construct stem function
    for kdx=1:self.N3
      for jdx=1:self.N2
        if (jdx==self.N2n)
         @simd for idx=1:self.N1
          psi0[idx,self.N2n,kdx]=(dt*v2n[pdx]).*psi0[idx,self.N2n,kdx]
         end
       else
         @simd for idx=1:self.N1
          psi0[idx,jdx,kdx]=(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx]).*DK2[jdx]
         end
       end
      end
    end

    v1n[pdx]+= q_m*eval_scalar(self, psi0 ,B3)
    v3n[pdx]+= (-q_m)*eval_scalar(self, psi0,B1)

    #Accumulation
    conj!(psi0)
    for kdx=1:self.N3
      for jdx=1:self.N2
       @simd for idx=1:self.N1
       J2[idx,jdx,kdx]+=psi0[idx,jdx,kdx]*w
       psi0[idx,jdx,kdx]=psi1[idx,jdx,kdx]
       end
      end
    end

    #Hp_3
    x3n[pdx]+=dt*v3n[pdx]
    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)
    # eval_fourier_modes2(dt*v3n[pdx]*pm.kx3, self.Nx3,psix3)
    # for kdx=1:self.N3
    #   for jdx=1:self.N2
    #     @simd for idx=1:self.N1
    #      psi1[idx,jdx,kdx]=psi0[idx,jdx,kdx]*psix3[kdx]
    #     end
    #   end
    # end


    #Construct stem function
    for kdx=1:self.N3
      if kdx==self.N3n
        for jdx=1:self.N2
          @simd for idx=1:self.N1
            psi0[idx,jdx,kdx]=(dt*v3n[pdx]).*psi0[idx,jdx,kdx]
          end
        end
      else
        for jdx=1:self.N2
          @simd for idx=1:self.N1
            psi0[idx,jdx,kdx]=(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx]).*DK3[kdx]
          end
        end
      end
    end

    v1n[pdx]+= (-q_m)*eval_scalar(self, psi0 ,B2)
    v2n[pdx]+= (q_m)*eval_scalar(self, psi0,B1)

    #Accumulation
    conj!(psi0)
    for kdx=1:self.N3
      for jdx=1:self.N2
       @simd for idx=1:self.N1
       J3[idx,jdx,kdx]+=psi0[idx,jdx,kdx]*w
       psi0[idx,jdx,kdx]=psi1[idx,jdx,kdx]
       end
      end
    end

   end
end


#
# function push_Hp1{T}(self::pm_pif3d, x1::T, x2::T, x3::T, v1::T, v2::T , v3::T, q::T, m::T, w::T,
#             psi0::Array{Complex{T},3},  dt::T,
#             J1::Array{Complex{T},3},B2::Array{Complex{T},3},B3::Array{Complex{T},3})
#
#     x1+=dt*v1
#     #psi1=copy(psi0).*reshape(eval_fourier_modes0(dt*v1*2*pi/self.L1,self.Nx1),
#                               #  self.N1,1,1)
#     psi1=Array{Complex{T}}(self.N1,self.N2,self.N3)
#     eval_basis(self,x1,x2,x3,psi1)
#     K=getK(self,1)
#     K=1./K
#     K[1]=0.0
#     psi0.=(psi1-psi0).*(-reshape(K,self.N1,1,1)*im)
#     psi0[1,:,:]=(dt*v1).*psi1[1,:,:]
#
#      v2+= (-q/m)*eval_scalar(self, psi0 ,B3)
#      v3+= (q/m)*eval_scalar(self, psi0,B2)
#
#     conj!(psi0)
#     psi0*=w*q/self.L1/self.L2/self.L3
#     J1.+=psi0
#
#     psi0[:]=psi1[:]
#     return x1,v2,v3
# end
#
#
# function push_Hp1{T}(self::pm_pif3d, x1::T, x2::T, x3::T, v1::T, v2::T , v3::T, q::T, m::T, w::T,
#             psi0::Array{Complex{T},3}, psi1::Array{Complex{T},3}, dt::T,
#             J1::Array{Complex{T},3},B2::Array{Complex{T},3},B3::Array{Complex{T},3})
#
#     x1+=dt*v1
#     psi1=copy(psi0).*reshape(eval_fourier_modes0(dt*v1*2*pi/self.L1,self.Nx1),
#                                self.N1,1,1)
#     psi1=Array{Complex{T}}(self.N1,self.N2,self.N3)
#     eval_basis(self,x1,x2,x3,psi1)
#     K=getK(self,1)
#     K=1./K
#     K[1]=0.0
#     psi0.=(psi1-psi0).*(-reshape(K,self.N1,1,1)*im)
#     psi0[1,:,:]=(dt*v1).*psi1[1,:,:]
#
#     v2+= (-q/m)*eval_scalar(self, psi0 ,B3)
#     v3+= (q/m)*eval_scalar(self, psi0,B2)
#     conj!(psi0)
#     psi0*=w*q/self.L1/self.L2/self.L3
#     J1.+=psi0
#
#     psi0[:]=psi1[:]
#     return x1,v2,v3
# end



  function push_Hp1{T}(self::pm_pif3d,

  x1::T, x2::T, x3::T, v1::T, v2::T , v3::T, q::T, m::T, w::T,
            psi0::Array{Complex{T},3}, psi1::Array{Complex{T},3}, dt::T,
            J1::Array{Complex{T},3},B2::Array{Complex{T},3},B3::Array{Complex{T},3})

    x1+=dt*v1
    psi1=copy(psi0).*reshape(eval_fourier_modes0(dt*v1*2*pi/self.L1,self.Nx1),
                               self.N1,1,1)
    psi1=Array{Complex{T}}(self.N1,self.N2,self.N3)
    eval_basis(self,x1,x2,x3,psi1)
    K=getK(self,1)
    K = 1. / K
    K[1] = 0.0
    psi0 .= (psi1-psi0).*(-reshape(K,self.N1,1,1)*im)
    psi0[1,:,:] .= (dt*v1) .* psi1[1,:,:]

    v2+= (-q/m)*eval_scalar(self, psi0 ,B3)
    v3+= (q/m)*eval_scalar(self, psi0,B2)
    conj!(psi0)
    psi0*=w*q/self.L1/self.L2/self.L3
    J1.+=psi0

    psi0[:]=psi1[:]
    return x1,v2,v3
end



function push_Hp2{T}(self::pm_pif3d, x1::T, x2::T, x3::T, v1::T, v2::T , v3::T, q::T, m::T, w::T,
            psi0::Array{Complex{T},3},  dt::T,
            J2::Array{Complex{T},3},B1::Array{Complex{T},3},B3::Array{Complex{T},3})

    x2+=dt*v2
    psi1=Array{Complex{T}}(self.N1,self.N2,self.N3)
    eval_basis(self,x1,x2,x3,psi1)
    K = getK(self,2)
    K = 1. / K
    K[self.Nx2+1]=0.0
    psi0.=(psi1-psi0).*(-reshape(K,1,self.N2,1)*im)
    psi0[:,self.Nx2+1,:]=(dt*v2).*psi1[:,self.Nx2+1,:]

    v1+= (q/m)*eval_scalar(self, psi0 ,B3)
    v3+= (-q/m)*eval_scalar(self, psi0,B1)
    J2.+=conj(  psi0 )*(w*q/self.L1/self.L2/self.L3)

    psi0[:]=psi1[:]
    return x2,v1,v3
end
function push_Hp3{T}(self::pm_pif3d, x1::T, x2::T, x3::T, v1::T, v2::T , v3::T, q::T, m::T, w::T,
            psi0::Array{Complex{T},3},  dt::T,
            J3::Array{Complex{T},3},B1::Array{Complex{T},3},B2::Array{Complex{T},3})

    x3+=dt*v3
    psi1=Array{Complex{T}}(self.N1,self.N2,self.N3)
    eval_basis(self,x1,x2,x3,psi1)
    K=getK(self,3)
    K = 1. / K
    K[self.Nx3+1]=0.0
    psi0.=(psi1-psi0).*(-reshape(K,1,1,self.N3)*im)
    psi0[:,:,self.Nx3+1]=(dt*v2).*psi1[:,:,self.Nx3+1]

    v1+= (-q/m)*eval_scalar(self, psi0 ,B2)
    v2+= (q/m)*eval_scalar(self, psi0,B1)
    J3.+=conj(  psi0 )*(w*q/self.L1/self.L2/self.L3)

    psi0[:]=psi1[:]
    return x3,v1,v2
end




#rho from particles
@inbounds function accum_osde2{T <:AbstractFloat}(self::pm_pif3d{T},
           x1n::Array{T,1},x2n::Array{T,1},  x3n::Array{T,1},
           wn::Array{T,1})
  Np::Int=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)
  @assert Np==length(wn)


  rhs=zeros(Complex{T},self.N1,self.N2,self.N3)
  psin=Array{Complex{T}}(self.N1,self.N2,self.N3)

  for pdx=1:Np

    #Julia is column major
    eval_basis(pm,x1n[pdx],x2n[pdx],x3n[pdx],psin)
    for kdx=1:self.N3
      for jdx=1:self.N2
        @simd for idx=1:self.N1
          rhs[idx,jdx,kdx]+=psin[idx,jdx,kdx]*wn[pdx]
        end
      end
    end
    #accum_basis(pm,psin,wn[pdx], rhs)
  end
  return conj(rhs)/(self.L1*self.L2*self.L3)
end


#rho from particles
function accum_osde{T <:AbstractFloat}(self::pm_pif3d{T},
           x1n::Array{T,1},x2n::Array{T,1},  x3n::Array{T,1},
           wn::Array{T,1})
  Np::Int=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)
  @assert Np==length(wn)

  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)

  rhs=zeros(Complex{T},self.N1,self.N2,self.N3)

  k1::T=2*pi/self.L1
  k2::T=2*pi/self.L2
  k3::T=2*pi/self.L3

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
  end
  end
  return conj(rhs)/(self.L1*self.L2*self.L3)
end


#Variance of rho from particles, given a rhs
function accum_osde_var{T}(self::pm_pif3d{T},
           rhs::Array{Complex{T},3}, # provide a mean
           x1n::Array{T,1},x2n::Array{T,1},  x3n::Array{T,1},
           wn::Array{T,1})
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)
  @assert Np==length(wn)

  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)

  rhs_var=zeros(T,self.N1,self.N2,self.N3)

  k1::T = -2*pi / self.L1
  k2::T = -2*pi / self.L2
  k3::T = -2*pi / self.L3

  vol::T = 1. / (self.L1*self.L2*self.L3)
  @inbounds begin
  for pdx=1:Np
    #Julia is column major
    eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)

    #Introduce weight
    @simd for idx=1:self.N1
    psi1[idx]*=wn[pdx]*vol
    end

    for kdx=1:self.N3
       for jdx=1:self.N2
        psi23::Complex{T}=psi2[jdx]*psi3[kdx]
        @simd  for idx=1:self.N1
           rhs_var[idx,jdx,kdx]+=real(conj(psi1[idx]*psi23-rhs[idx,jdx,kdx])*
                                      (psi1[idx]*psi23-rhs[idx,jdx,kdx]))
        end
      end
    end
  end
  end
  return rhs_var
end

# Accumulate on accum_psin
@inbounds function accum_basis{T}(self::pm_pif3d{T}, psin::Array{Complex{T},3},
            weight::T, accum_psin::Array{Complex{T},3})
  for kdx=1:self.N3
    for jdx=1:self.N2
      @simd for idx=1:self.N1
        accum_psin[idx,jdx,kdx]+=psin[idx,jdx,kdx]*weight
      end
    end
  end
end
#rho for indefinite integration
function accum_osde_int{T <:AbstractFloat}(self::pm_pif3d{T},
           x1n::Array{T,1},x2n::Array{T,1},  x3n::Array{T,1},
           wn::Array{T,1}, axis::Int)
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)
  @assert Np==length(wn)

  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)

  rhs=zeros(Complex{T},self.N1,self.N2,self.N3)

  k1=2*pi/self.L1
  k2=2*pi/self.L2
  k3=2*pi/self.L3

  K1,K2,K3=getK(self)
  K1[1]=1.0; K2[self.Nx2+1]=1.0; K3[self.Nx3+1]=1.0
  #K1=1./K1; K2=1./K2; K3=1./K3;



  @inbounds begin
  for pdx=1:Np
    #Julia is column major
    eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)

      if (axis==1)
       #psi1.*=K1*(-im)
       psi1*=(-im)
       psi1[1]=x1n[pdx]
      elseif (axis==2)
       #psi2.*=K2*(-im)
       psi2[self.Nx2+1]=x2n[pdx]
     elseif (axis==3)
       #psi3.*=K3*(-im)
       psi3[self.Nx3+1]=x3n[pdx]
     end

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
  end
  end

  if (axis==1)
   rhs=rhs./reshape(K1,self.N1,1,1)
  elseif (axis==2)
   #psi2.*=K2*(-im)
  #  psi2[self.Nx2+1]=x2n[pdx]
  elseif (axis==3)
   #psi3.*=K3*(-im)
  #  psi3[self.Nx3+1]=x3n[pdx]
  end


  return conj(rhs)/self.L1/self.L2/self.L3
end
#
# function rhs_particle2{T <:AbstractFloat}(self::pm_pif3d,
#            x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},  wn::Array{T,1})
#   Np=length(x1n)
#   @assert Np==length(x2n)
#   @assert Np==length(x3n)
#   @assert Np==length(wn)
#
#   psi1=Array{Complex{T}}(self.Nx1)
#   psi2=Array{Complex{T}}(self.Nx2)
#   psi3=Array{Complex{T}}(self.Nx3)
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
#     eval_fourier_modes(x1n[pdx]*k1, self.Nx1,psi1)
#     eval_fourier_modes(x2n[pdx]*k2, self.Nx2,psi2)
#     eval_fourier_modes(x3n[pdx]*k3, self.Nx3,psi3)
#
#     #Introduce weight
#     @simd for idx=1:self.N1
#     psi1[idx]*=wn[pdx]
#     end
#     # for kdx=1:self.Nx1
#     #
#     #     for jdx=1:self.Nx2
#     #     psi23::Complex{T}=psi2[jdx]*psi3[kdx]
#     #
#     #
#     #
#     #     end
#     #
#     #
#     #
#     #
#     #
#     # end
#     #
#     # for kdx=1:self.N3
#     #    for jdx=1:self.N2
#     #     psi23::Complex{T}=psi2[jdx]*psi3[kdx]
#     #     @simd  for idx=1:self.N1
#     #        rhs[idx,jdx,kdx]+=psi1[idx]*psi23
#     #     end
#     #   end
#     # end
#
#   end
#   end
#   return conj(rhs)/self.L1/self.L2/self.L3
# end


function eval_scalar{T}(self::pm_pif3d,
           x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
           F::Array{Complex{T},3},Fn::Array{Complex{T},1})
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)

  #Fn=zeros(Complex{T},Np)
  Fn[:]=0
  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)
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
        psi23*=2.
        @simd  for idx=2:self.N1
          psin::Complex{T}=psi1[idx]*psi23
           Fn[pdx]+=(real(psin)*real(F[idx,jdx,kdx])
                    -imag(psin)*imag(F[idx,jdx,kdx]))
        end
      end
    end

  end
  nothing
end


@inbounds function eval_scalar{T}(self::pm_pif3d,
           x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
           F::Array{Complex{T},3},Fn::Array{T,1})
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)

  #Fn=Array(T,Np)
  #Fn[:]=0
  psi1=Array{Complex{T}}(self.Nx1)
  psi2=Array{Complex{T}}(self.Nx2)
  psi3=Array{Complex{T}}(self.Nx3)
  k1=2*pi/self.L1
  k2=2*pi/self.L2
  k3=2*pi/self.L3
  # Inidicies of zero modes
  N1n=1
  N2n=self.Nx2+1
  N3n=self.Nx3+1


  for pdx=1:Np
    eval_fourier_modes(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes(x3n[pdx]*k3, self.Nx3,psi3)


    # k3=0,k2=0,k1=0
    Fn[pdx]=real(F[1,N2n,N3n])
    # k3=0,k2=0, k1!=0
    @simd  for idx=1:self.Nx1
     Fn[pdx]+=sumcnj(F[1+idx,N2n,N3n], psi1[idx] )
    end
    # k3=0, k2!=0, k1
    @simd for jdx=1:self.Nx2
      # k3=0, k2!=0, k1=0
      Fn[pdx]+=real( F[1,N2n+jdx,N3n]*psi2[jdx]
                    +F[1,N2n-jdx,N3n]*conj(psi2[jdx]) )
    end
    for jdx=1:self.Nx2
      # k3=0, k2!=0, k1
      @simd  for idx=1:self.Nx1
        Fn[pdx]+=(sumcnj(F[1+idx,N2n+jdx,N3n], psi2[jdx]*psi1[idx] )
                 +sumcnj(F[1+idx,N2n-jdx,N3n], conj(psi2[jdx])*psi1[idx]) )
      end
    end

    #k3!=0
    @simd for kdx=1:self.Nx3
      #k3!=0, k2=0, k1=0
      Fn[pdx]+=real( F[1,N2n,N3n+kdx]*psi3[kdx]
               +F[1,N2n,N3n-kdx]*conj(psi3[kdx]) )
    end

    for kdx=1:self.Nx3
      @simd  for idx=1:self.Nx1
        Fn[pdx]+=( sumcnj(F[1+idx,N2n,N3n+kdx], psi3[kdx]*psi1[idx] )
                 +sumcnj(F[1+idx,N2n,N3n-kdx], conj(psi3[kdx])*psi1[idx]) )
      end

      for jdx=1:self.Nx2
        psi23::Complex{T}=psi2[jdx]*psi3[kdx]
        psi23_::Complex{T}=conj(psi2[jdx])*psi3[kdx]
        Fn[pdx]+=real( F[1,N2n+jdx,N3n+kdx]*psi23
        +F[1,N2n-jdx,N3n-kdx]*conj(psi23)
        +F[1,N2n+jdx,N3n-kdx]*conj(psi23_)
        +F[1,N2n-jdx,N3n+kdx]*psi23_)
        @simd  for idx=1:self.Nx1
          Fn[pdx]+=(sumcnj(F[1+idx,N2n+jdx,N3n+kdx], psi23*psi1[idx] )
          +sumcnj(F[1+idx,N2n-jdx,N3n-kdx], conj(psi23)*psi1[idx])
          +sumcnj(F[1+idx,N2n+jdx,N3n-kdx],conj(psi23_)*psi1[idx] )
          +sumcnj(F[1+idx,N2n-jdx,N3n+kdx],psi23_*psi1[idx] ) )
        end
      end
      # for jdx=1:self.Nx2
      #   Fn[pdx]+=real( F[1,N2n+jdx,N3n+kdx]*psi2[jdx]*psi3[kdx]
      #   +F[1,N2n-jdx,N3n-kdx]*conj(psi2[jdx]*psi3[kdx])
      #   +F[1,N2n+jdx,N3n-kdx]*psi2[jdx]*conj(psi3[kdx])
      #   +F[1,N2n-jdx,N3n+kdx]*conj(psi2[jdx])*psi3[kdx] )
      #   @simd  for idx=1:self.Nx1
      #     Fn[pdx]+=(sumcnj(F[1+idx,N2n+jdx,N3n+kdx], psi2[jdx]*psi3[kdx]*psi1[idx] )
      #     +sumcnj(F[1+idx,N2n-jdx,N3n-kdx], conj(psi2[jdx]*psi3[kdx])*psi1[idx])
      #     +sumcnj(F[1+idx,N2n+jdx,N3n-kdx], psi2[jdx]*conj(psi3[kdx])*psi1[idx] )
      #     +sumcnj(F[1+idx,N2n-jdx,N3n+kdx],conj(psi2[jdx])*psi3[kdx]*psi1[idx] ) )
      #   end
      # end

    end



  end
  nothing
end





function eval_vectorfield{T}(self::pm_pif3d,
           x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
           F1::Array{Complex{T},3},F2::Array{Complex{T},3},F3::Array{Complex{T},3},
           F1n::Array{Complex{T},1},F2n::Array{Complex{T},1},F3n::Array{Complex{T},1})
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)
  #Fn=zeros(Complex{T},Np)
  F1n[:]=0.0
  F2n[:]=0
  F3n[:]=0.0
  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)
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
        F1n[pdx]+=psi23*F1[1,jdx,kdx]
        F2n[pdx]+=psi23*F2[1,jdx,kdx]
        F3n[pdx]+=psi23*F3[1,jdx,kdx]
        psi23*=2
        for idx=2:self.N1
            psin::Complex{T}=psi1[idx]*psi23
            F1n[pdx]+=(real(psin)*real(F1[idx,jdx,kdx])
                    -imag(psin)*imag(F1[idx,jdx,kdx]))
            F2n[pdx]+=(real(psin)*real(F2[idx,jdx,kdx])
                   -imag(psin)*imag(F2[idx,jdx,kdx]))
            F3n[pdx]+=(real(psin)*real(F3[idx,jdx,kdx])
                    -imag(psin)*imag(F3[idx,jdx,kdx]))
        end
      end
    end

  end

end

@inbounds function eval_scalar{T}(self::pm_pif3d,
           x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
           F::Array{Array{Complex{T},3},1}, Fn::Array{Array{Complex{T},1},1})
  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)
  D::Int=length(F)

  @simd for ddx=1:D
    Fn[ddx][:].=0.0
  end

  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)
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
          Fn[ddx][pdx].+=psi23*F[ddx][1,jdx,kdx]
        end
        for ddx=1:D
          @simd  for idx=2:self.N1
              psin::Complex{T}=psi1[idx]*psi23

              Fn[ddx][pdx]+=(real(psin)*real(F[ddx][idx,jdx,kdx])
                          -imag(psin)*imag(F[ddx][idx,jdx,kdx]))*2
              end
            end

      end
    end

  end
  nothing
end



function getK(self::pm_pif3d,axis::Int)
  if axis==1
  K=collect(0:self.Nx1)/self.L1*2*pi
  elseif axis==2
  K=collect(-self.Nx2:self.Nx2)/self.L2*2*pi
  else
  K=collect(-self.Nx3:self.Nx3)/self.L3*2*pi
  end

  return K
end
function getK(self::pm_pif3d)
  K1=collect(0:self.Nx1)/self.L1*2*pi
  K2=collect(-self.Nx2:self.Nx2)/self.L2*2*pi
  K3=collect(-self.Nx3:self.Nx3)/self.L3*2*pi
  return K1,K2,K3
end

function getK2(self::pm_pif3d)
  K1,K2,K3=getK(self)
  K1=reshape(K1,self.N1,1,1)
  K2=reshape(K2,1,self.N2,1)
  K3=reshape(K3,1,1,self.N3)
  return K1,K2,K3
end






function solve_poisson!{T}(self::pm_pif3d,
                      rho::Array{Complex{T},3}, Phi::Array{Complex{T},3})
    K1,K2,K3=getK(self)
     for kdx=1:self.N3
        for jdx=1:self.N2
          for idx=1:self.N1
           if !(idx==1 && jdx==self.Nx2+1  && kdx==self.Nx3+1 )

            Phi[idx,jdx,kdx]=rho[idx,jdx,kdx]/(K1[idx]^2 +K2[jdx]^2 +K3[kdx]^2)
          else
            Phi[idx,jdx,kdx]=0.
          end
         end
       end
     end
     return Phi
end
function solve_poisson{T}(self::pm_pif3d{T}, rho::Array{Complex{T},3})
  Phi=Array{Complex{T}}(self.N1,self.N2,self.N3)
  solve_poisson!(self,rho,Phi)
  return Phi
end


function solve_gauss{T}(self::pm_pif3d{T}, rho::Array{Complex{T},3} )
  E1=similar(rho)
  E2=similar(rho)
  E3=similar(rho)
  solve_gauss!(self, rho,E1,E2,E3)
  return E1,E2,E3
end

function gauss_error{T}(self::pm_pif3d{T}, rho::Array{Complex{T},3}, E1,E2,E3 )
  K1,K2,K3=getK2(self)
  rho2=similar(rho)
  rho2[:]=rho[:]
  rho2[1,self.Nx2+1,self.Nx3+1]=0.0
  return maximum(abs.(E1.*(im*K1)+E2.*(im*K2)+E3.*(im*K3)-rho2))
 end

function solve_gauss!{T}(self::pm_pif3d{T}, rho::Array{Complex{T},3}, E1,E2,E3   )
  Phi=solve_poisson(self,rho)
  K1,K2,K3=getK2(self)
  E1::Array{Complex{T},3}=-Phi*im.*K1
  E2::Array{Complex{T},3}=-Phi*im.*K2
  E3::Array{Complex{T},3}=-Phi*im.*K3
end


import Base.LinAlg: gradient

# Gradient of a scalar function
function gradient{T}(self::pm_pif3d{T}, F::Array{Complex{T},3})
  K1,K2,K3=getK2(self)
  DF1::Array{Complex{T},3}=(F*im).*K1
  DF2::Array{Complex{T},3}=(F*im).*K2
  DF3::Array{Complex{T},3}=(F*im).*K3
  return DF1,DF2,DF3
end

import Base: div
function div{T}(self::pm_pif3d{T}, F::Array{Complex{T},3})
  DF1,DF2,DF3=gradient(pm_pif3d, F)
  return DF1.+DF2.+DF3
end



function curl{T}(self::pm_pif3d, F1::Array{Complex{T},3},
              F2::Array{Complex{T},3},F3::Array{Complex{T},3} )
    K1,K2,K3=getK2(self)
    return im*(K2.*F3 - K3.*F2), im*(K3.*F1 - K1.*F3), im*(K1.*F2 - K2.*F1)

end


function H1seminorm{T}(self::pm_pif3d{T},F::Array{Complex{T},3} )
  K1,K2,K3=getK2(self)
  return 2.0*sum((abs.(F.*K1).^2+abs.(F.*K2).^2
             +abs.(F.*K2).^2)[:])*self.L1*self.L2*self.L3
end


function L2norm{T}(self::pm_pif3d{T},F::Array{Complex{T},3}  )
  L2=sum(abs.(F[:]).^2)*2.0- sum((abs.(F[self.N1n,:,:]).^2)[:])
  L2*=self.L1*self.L2*self.L3
  return L2
end

import Base.LinAlg: dot
function dot{T}(self::pm_pif3d{T}, A::Array{Complex{T},3},B::Array{Complex{T},3})
    ab=(sum(real(conj(A[2:end,:,:])).*real(B[2:end,:,:]))-sum(imag(conj(A[2:end,:,:])).*imag(B[2:end,:,:])))*2
          +imag(sum(conj(A[self.N1n,:,:]).*B[self.N1n,:,:]))
    return ab*self.L1*self.L2*self.L3
end


# Integrate Ampere equation from 0 to dt
function integrate_H_B{T}(self::pm_pif3d{T}, E1::Array{Complex{T},3},
           E2::Array{Complex{T},3},E3::Array{Complex{T},3},
           B1::Array{Complex{T},3},B2::Array{Complex{T},3},
           B3::Array{Complex{T},3} ,c::T, dt::T)

  F1,F2,F3=curl(self, B1,B2,B3)
  E1.+= dt*F1*c^2
  E2.+= dt*F2*c^2
  E3.+= dt*F3*c^2
  return E1,E2,E3
end


# # Integrate Ampere equation from 0 to dt
# function integrate_H_B_E1{T}(self::pm_pif3d{T}, E1::Array{Complex{T},3},
#            B1::Array{Complex{T},3},B2::Array{Complex{T},3},
#            B3::Array{Complex{T},3} ,c::T, dt::T,)
#
#   F1,F2,F3=curl(self, B1,B2,B3)
#   E1.+= dt*c^2* (       )
# end



# Integrate Ampere equation from 0 to dt
function integrate_Faraday{T}(self::pm_pif3d{T}, E1::Array{Complex{T},3},
           E2::Array{Complex{T},3},E3::Array{Complex{T},3},
           B1::Array{Complex{T},3},B2::Array{Complex{T},3},
           B3::Array{Complex{T},3} ,dt::T)
  F1,F2,F3=curl(self, E1,E2,E3)
  B1.-=dt*F1
  B2.-=dt*F2
  B3.-=dt*F3
end


@inbounds function integrate_H_E{T}(self::pm_pif3d{T}, E1::Array{Complex{T},3},
           E2::Array{Complex{T},3},E3::Array{Complex{T},3},
           B1::Array{Complex{T},3},B2::Array{Complex{T},3},B3::Array{Complex{T},3},
           x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
           v1n::Array{T,1},v2n::Array{T,1}, v3n::Array{T,1},
           qn::Array{T,1}, mn::Array{T,1}, dt::T)

  integrate_H_E(self,E1,E2,E3,x1n,x2n,x3n,v1n,v2n,v3n,qn,mn,dt)

  integrate_Faraday(self,E1,E2,E3,B1,B2,B3,dt)
end


#
@inbounds function integrate_H_E{T}(self::pm_pif3d{T}, E::Array{Complex{T},3},
                              x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
                        vn::Array{T,1},qn::Array{T,1}, mn::Array{T,1},
                                         dt::T)

  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)

  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)
  k1::T=2*pi/self.L1
  k2::T=2*pi/self.L2
  k3::T=2*pi/self.L3
  #psi0=Array{Complex{T}}(self.N1,self.N2,self.N3)
  for pdx=1:Np
    eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)
    En::Complex{T}=0.0
    for kdx=1:self.N3
       for jdx=1:self.N2
        psi23::Complex{T}=psi2[jdx]*psi3[kdx]
        En+=psi23*E[1,jdx,kdx]
        psi23*=2
        @simd for idx=2:self.N1
            psin::Complex{T}=psi1[idx]*psi23
            En+=(real(psin)*real(E[idx,jdx,kdx])
                    -imag(psin)*imag(E[idx,jdx,kdx]))
        end
      end
    end
    dtqm::T=dt*qn[pdx]./mn[pdx]
    vn[pdx]+=dtqm*real(En)
  end

end



@inbounds function integrate_H_E{T}(self::pm_pif3d{T}, E1::Array{Complex{T},3},
                          E2::Array{Complex{T},3},  E3::Array{Complex{T},3},
                              x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
                        v1n::Array{T,1},v2n::Array{T,1}, v3n::Array{T,1},
                              qn::Array{T,1}, mn::Array{T,1},
                                         dt::T)

  Np=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)
  #Fn=zeros(Complex{T},Np)

  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)
  k1::T=2*pi/self.L1
  k2::T=2*pi/self.L2
  k3::T=2*pi/self.L3
  #psi0=Array{Complex{T}}(self.N1,self.N2,self.N3)
  for pdx=1:Np
    eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)
    E1n::Complex{T}=0.0
    E2n::Complex{T}=0.0
    E3n::Complex{T}=0.0
    for kdx=1:self.N3
       for jdx=1:self.N2
        psi23::Complex{T}=psi2[jdx]*psi3[kdx]
        E1n+=psi23*E1[1,jdx,kdx]
        E2n+=psi23*E2[1,jdx,kdx]
        E3n+=psi23*E3[1,jdx,kdx]
        psi23*=2
        @simd for idx=2:self.N1
            psin::Complex{T}=psi1[idx]*psi23
            E1n+=(real(psin)*real(E1[idx,jdx,kdx])
                    -imag(psin)*imag(E1[idx,jdx,kdx]))
            E2n+=(real(psin)*real(E2[idx,jdx,kdx])
                   -imag(psin)*imag(E2[idx,jdx,kdx]))
            E3n+=(real(psin)*real(E3[idx,jdx,kdx])
                    -imag(psin)*imag(E3[idx,jdx,kdx]))
        end
      end
    end
    dtqm::T=dt*qn[pdx]./mn[pdx]
    v1n[pdx]+=dtqm*real(E1n)
    v2n[pdx]+=dtqm*real(E2n)
    v3n[pdx]+=dtqm*real(E3n)

    # eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi0)
    # dtqm::T=dt*qn[pdx]./mn[pdx]
    # v1n[pdx]+=dtqm*eval_scalar(self, psi0 ,E1)
    # v2n[pdx]+=dtqm*eval_scalar(self, psi0 ,E2)
    # v3n[pdx]+=dtqm*eval_scalar(self, psi0 ,E3)
  end

end



function test_eval_basis{T}(pm::pm_pif3d{T}, N::Int )
  psin=zeros(Complex{Float64},pm.N1,pm.N2,pm.N3 )
  psin2=zeros(Complex{Float64},pm.N1,pm.N2,pm.N3 )
  maxerr::T=0.0
  for idx=1:N
  x=rand(T,3)
  eval_basis(pm,x...,psin)
  eval_basis2(pm,x...,psin2)
  err=maximum(abs.(psin-psin2))
  if ( err > maxerr)
    maxerr=err
  end
  end
  return maxerr
end


function integrate_Hp_boris_exp{T}(self::pm_pif3d{T},x1n::Array{T,1},x2n::Array{T,1},
                       x3n::Array{T,1},
                       v1n::Array{T,1},v2n::Array{T,1}, v3n::Array{T,1},
                      qn::Array{T,1}, mn::Array{T,1},wn::Array{T,1},
          J1::Array{Complex{T},3},J2::Array{Complex{T},3},J3::Array{Complex{T},3},
          B1::Array{Complex{T},3},B2::Array{Complex{T},3}, B3::Array{Complex{T},3},
          dtA::T,dtB::T,async::Bool=false)

          return integrate_Hp_boris_exp(self, x1n,x2n,x3n,
                       v1n,v2n,v3n,qn,mn,wn,
                       J1,J2,J3,B1,B2,B3,[dtA],[dtB],async)
end


function integrate_Hp_boris_exp{T}(self::pm_pif3d{T},x1n::Array{T,1},x2n::Array{T,1},
                       x3n::Array{T,1},
                       v1n::Array{T,1},v2n::Array{T,1}, v3n::Array{T,1},
                      qn::Array{T,1}, mn::Array{T,1},wn::Array{T,1},
          J1::Array{Complex{T},3},J2::Array{Complex{T},3},J3::Array{Complex{T},3},
          B1::Array{Complex{T},3},B2::Array{Complex{T},3}, B3::Array{Complex{T},3},
          dtA::Array{T,1},dtB::Array{T,1},async::Bool=false)
  Np::Int=length(x1n)
  @assert length(dtA)==length(dtB)
  Ndt::Int=length(dtA)

  psi1=Array{Complex{T}}(self.N1,self.N2,self.N3)
  psi0=Array{Complex{T}}(self.N1,self.N2,self.N3)


  K1,K2,K3=getK(self)

  vol::T = 1. / (self.L1*self.L2*self.L3)



  for pdx=1:Np

    w::T=wn[pdx]*qn[pdx]*vol #Accumulated weight with fourier normalization
    psi123::Complex{T}=0.#
    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi0)
    # Rotation in magnetic field
    v1n[pdx],v2n[pdx],v3n[pdx]=integrate_vxB(v1n[pdx],v2n[pdx],v3n[pdx],
                            eval_scalar(self, psi0 ,B1),
                            eval_scalar(self, psi0 ,B2),
                            eval_scalar(self, psi0 ,B3),
                             qn[pdx],mn[pdx],dtA[1])

    for sdx=1:Ndt
      ################################
      dt::T=dtA[sdx]+dtB[sdx]

      x1n[pdx]+=dt*v1n[pdx]
      x2n[pdx]+=dt*v2n[pdx]
      x3n[pdx]+=dt*v3n[pdx]
      eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)

        for kdx=1:self.N3
          kv3::T=K3[kdx]*v3n[pdx]
          for jdx=1:self.N2
            kv23::T=K2[jdx]*v2n[pdx]+kv3
           for idx=1:self.N1
            kv123::T=K1[idx]*v1n[pdx]+kv23
            if kv123==0. # DIRTY
              psi123=dt*w*conj(psi0[idx,jdx,kdx])
              else
              psi123=im*conj(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx])/kv123*w
            end
            J1[idx,jdx,kdx]+=v1n[pdx]*psi123
            J2[idx,jdx,kdx]+=v2n[pdx]*psi123
            J3[idx,jdx,kdx]+=v3n[pdx]*psi123
            psi0[idx,jdx,kdx]=psi1[idx,jdx,kdx]

           end
          end
        end
       ######################

       if (sdx==Ndt)
           if async
               dt=0
           else
               dt=dtB[Ndt]
           end
       else
         dt=dtB[sdx]+dtA[sdx+1]
       end

        if (dt!=0)
        # Rotation in magnetic field
        v1n[pdx],v2n[pdx],v3n[pdx]=integrate_vxB(v1n[pdx],v2n[pdx],v3n[pdx],
                                eval_scalar(self, psi0 ,B1),
                                eval_scalar(self, psi0 ,B2),
                                eval_scalar(self, psi0 ,B3),
                                 qn[pdx],mn[pdx],dt)
        end

      end

  end

  # not common since function only accumulates charge on given J
  # conj!(J1)
  # conj!(J2)
  # conj!(J3)
end


function integrate_vxB{T}(v1::T,v2::T,v3::T,b1::T,b2::T,b3::T,q::T,m::T,dt::T)
  b::T=sqrt(b1^2 + b2.^2 + b3^2) # Norm of magnetic field

  if b==0
    return v1,v2,v3
  else

    a::T=b*(q./m)*dt
    cisa::Complex{T}=cis(a)
    cosa::T=real(cisa)
    sina::T=imag(cisa)

    b1=b1/b
    b2=b2/b
    b3=b3/b


    v1_ =(b1^2 * (1 - cosa) + cosa) * v1 +
         (b1  .* b2  * (1 -cosa) + b3.*sina) .* v2 +
         (b1  .* b3 .* (1-cosa)  - b2.*sina) .* v3

    v2_ = (b1 .* b2 .* (1 - cosa) - b3.*sina) .* v1 +
          (b2 .^2   .* (1 - cosa) + cosa)     .* v2 +
          (b2 .* b3 .* (1 - cosa) + b1.*sina) .* v3

    v3_ = (b1 * b3 * (1 - cosa) + b2 * sina) * v1 +
          (b3 * b3 * (1 - cosa) - b1 * sina) * v2 +
          (b3^2 * (1 - cosa) + cosa) * v3

  return v1_, v2_, v3_

  end
end

@inbounds function integrate_vxB{T}(self::pm_pif3d{T}, B1::Array{Complex{T},3},
                          B2::Array{Complex{T},3},  B3::Array{Complex{T},3},
                              x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
                        v1n::Array{T,1},v2n::Array{T,1}, v3n::Array{T,1},
                              qn::Array{T,1}, mn::Array{T,1},
                                         dt::T)

  Np::Int=length(x1n)
  @assert Np==length(x2n)
  @assert Np==length(x3n)

  psi1=Array{Complex{T}}(self.N1)
  psi2=Array{Complex{T}}(self.N2)
  psi3=Array{Complex{T}}(self.N3)
  k1::T=2*pi/self.L1
  k2::T=2*pi/self.L2
  k3::T=2*pi/self.L3
  for pdx=1:Np
    eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
    eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
    eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)
    B1n::Complex{T}=0.0
    B2n::Complex{T}=0.0
    B3n::Complex{T}=0.0
    for kdx=1:self.N3
       for jdx=1:self.N2
        psi23::Complex{T}=psi2[jdx]*psi3[kdx]
        B1n+=psi23*B1[1,jdx,kdx]
        B2n+=psi23*B2[1,jdx,kdx]
        B3n+=psi23*B3[1,jdx,kdx]
        psi23*=2
        @simd for idx=2:self.N1
            psin::Complex{T}=psi1[idx]*psi23
            B1n+=(real(psin)*real(B1[idx,jdx,kdx])
                    -imag(psin)*imag(B1[idx,jdx,kdx]))
            B2n+=(real(psin)*real(B2[idx,jdx,kdx])
                   -imag(psin)*imag(B2[idx,jdx,kdx]))
            B3n+=(real(psin)*real(B3[idx,jdx,kdx])
                    -imag(psin)*imag(B3[idx,jdx,kdx]))
        end
      end
    end
    v1n[pdx],v2n[pdx],v3n[pdx]=integrate_vxB(
                            v1n[pdx],v2n[pdx],v3n[pdx],
                        real(B1n),real(B2n),real(B3n),
                         qn[pdx],mn[pdx],dt)

  end

end




function integrate_Hp_split_sym{T}(self::pm_pif3d{T},x1n::Array{T,1},x2n::Array{T,1},
                       x3n::Array{T,1},
                       v1n::Array{T,1},v2n::Array{T,1}, v3n::Array{T,1},
                      qn::Array{T,1}, mn::Array{T,1},wn::Array{T,1},
          J1::Array{Complex{T},3},J2::Array{Complex{T},3},J3::Array{Complex{T},3},
          B1::Array{Complex{T},3},B2::Array{Complex{T},3}, B3::Array{Complex{T},3},
          dtA::Array{T,1},dtB::Array{T,1})
  Np::Int=length(x1n)
  @assert length(dtA)==length(dtB)

  psi1=Array{Complex{T}}(self.N1,self.N2,self.N3)
  psi0=Array{Complex{T}}(self.N1,self.N2,self.N3)

  psix1=Array{Complex{T}}(self.N1)
  psix2=Array{Complex{T}}(self.N2)
  psix3=Array{Complex{T}}(self.N3)

  #Fourier stem function Integral
  DK1=Array{Complex{T}}(self.N1)
  DK1[:]=getK(self,1)
  DK1 = 1. / (im*DK1)
  DK1[self.N1n]=0.0
  DK2 = Array{Complex{T}}(self.N2)
  DK2[:]=getK(self,2)
  DK2 = 1. / (im*DK2)
  DK2[self.N2n]=0.0
  DK3 = Array{Complex{T}}(self.N3)
  DK3[:] = getK(self,3)
  DK3 = 1. / (im*DK3)
  DK3[self.N3n] = 0.0

  vol::T = 1. / (self.L1*self.L2*self.L3)

  for pdx=1:Np
    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi0)
    w::T=wn[pdx]*qn[pdx]*vol #Accumulated weight with fourier normalization
    q_m::T=qn[pdx]/mn[pdx]

    #Hp_3
    x3n[pdx]+=dtA[1]*v3n[pdx]
    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)
    #Construct stem function
    for kdx=1:self.N3
      if kdx==self.N3n
        for jdx=1:self.N2
          @simd for idx=1:self.N1
            psi0[idx,jdx,kdx]=(dtA[1]*v3n[pdx]).*psi0[idx,jdx,kdx]
          end
        end
      else
        for jdx=1:self.N2
          @simd for idx=1:self.N1
            psi0[idx,jdx,kdx]=(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx]).*DK3[kdx]
          end
        end
      end
    end

    v1n[pdx]+= (-q_m)*eval_scalar(self, psi0 ,B2)
    v2n[pdx]+= (q_m)*eval_scalar(self, psi0,B1)

    #Accumulation
    conj!(psi0)
    for kdx=1:self.N3
      for jdx=1:self.N2
       @simd for idx=1:self.N1
       J3[idx,jdx,kdx]+=psi0[idx,jdx,kdx]*w
       psi0[idx,jdx,kdx]=psi1[idx,jdx,kdx]
       end
      end
    end

    for tdx=1:length(dtA)
      #Hp_2
      x2n[pdx]+=dtA[tdx]*v2n[pdx]
      eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)
      #Construct stem function
      for kdx=1:self.N3
        for jdx=1:self.N2
          if (jdx==self.N2n)
           @simd for idx=1:self.N1
            psi0[idx,self.N2n,kdx]=(dtA[tdx]*v2n[pdx]).*psi0[idx,self.N2n,kdx]
           end
         else
           @simd for idx=1:self.N1
            psi0[idx,jdx,kdx]=(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx]).*DK2[jdx]
           end
         end
        end
      end
      v1n[pdx]+= q_m*eval_scalar(self, psi0 ,B3)
      v3n[pdx]+= (-q_m)*eval_scalar(self, psi0,B1)
      #Accumulation
      conj!(psi0)
      for kdx=1:self.N3
        for jdx=1:self.N2
         @simd for idx=1:self.N1
         J2[idx,jdx,kdx]+=psi0[idx,jdx,kdx]*w
         psi0[idx,jdx,kdx]=psi1[idx,jdx,kdx]
         end
        end
      end


      #Hp_1
      x1n[pdx]+=(dtA[tdx]+dtB[tdx])*v1n[pdx]
      eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)

      #Construct stem function
      for kdx=1:self.N3
        for jdx=1:self.N2
          psi0[1,jdx,kdx]=((dtA[tdx]+dtB[tdx])*v1n[pdx]).*(psi0[1,jdx,kdx])
          @simd for idx=2:self.N1
           psi0[idx,jdx,kdx]=(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx]).*DK1[idx]
          end
        end
      end

      v2n[pdx]+= (-q_m)*eval_scalar(self, psi0 ,B3)
      v3n[pdx]+= q_m*eval_scalar(self, psi0,B2)

      #Accumulation
      conj!(psi0)
      for kdx=1:self.N3
        for jdx=1:self.N2
         @simd for idx=1:self.N1
         J1[idx,jdx,kdx]+=psi0[idx,jdx,kdx]*w
         psi0[idx,jdx,kdx]=psi1[idx,jdx,kdx]
         end
        end
      end

      #Hp_2
      x2n[pdx]+=dtB[tdx]*v2n[pdx]
      eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)
      #Construct stem function
      for kdx=1:self.N3
        for jdx=1:self.N2
          if (jdx==self.N2n)
           @simd for idx=1:self.N1
            psi0[idx,self.N2n,kdx]=(dtB[tdx]*v2n[pdx]).*psi0[idx,self.N2n,kdx]
           end
         else
           @simd for idx=1:self.N1
            psi0[idx,jdx,kdx]=(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx]).*DK2[jdx]
           end
         end
        end
      end
      v1n[pdx]+= q_m*eval_scalar(self, psi0 ,B3)
      v3n[pdx]+= (-q_m)*eval_scalar(self, psi0,B1)
      #Accumulation
      conj!(psi0)
      for kdx=1:self.N3
        for jdx=1:self.N2
         @simd for idx=1:self.N1
         J2[idx,jdx,kdx]+=psi0[idx,jdx,kdx]*w
         psi0[idx,jdx,kdx]=psi1[idx,jdx,kdx]
         end
        end
      end

      # Wrap around
      dt3::T=dtB[tdx]
      if tdx<length(dtA)
        dt3+=dtA[tdx+1]
      end

      #Hp_3
      x3n[pdx]+=dt3*v3n[pdx]
      eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)
      #Construct stem function
      for kdx=1:self.N3
        if kdx==self.N3n
          for jdx=1:self.N2
            @simd for idx=1:self.N1
              psi0[idx,jdx,kdx]=(dt3*v3n[pdx]).*psi0[idx,jdx,kdx]
            end
          end
        else
          for jdx=1:self.N2
            @simd for idx=1:self.N1
              psi0[idx,jdx,kdx]=(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx]).*DK3[kdx]
            end
          end
        end
      end

      v1n[pdx]+= (-q_m)*eval_scalar(self, psi0 ,B2)
      v2n[pdx]+= (q_m)*eval_scalar(self, psi0,B1)

      #Accumulation
      conj!(psi0)
      for kdx=1:self.N3
        for jdx=1:self.N2
         @simd for idx=1:self.N1
         J3[idx,jdx,kdx]+=psi0[idx,jdx,kdx]*w
         psi0[idx,jdx,kdx]=psi1[idx,jdx,kdx]
         end
        end
      end

    end
    #End of particle push
   end

   #conj!(J1)
   #conj!(J2)
   #conj!(J3)
end





# Cross product with matrices
function cross!{T}(v::Array{T,1},w::Array{T,2})
  N=size(w,2)
  @simd for idx=1:N
    w[:,idx]=cross(v,w[:,idx])
  end
  return w
end

function cross!{T}(w::Array{T,2},v::Array{T,1})
  N=size(w,2)
  @simd for idx=1:N
    w[:,idx]=cross(w[:,idx],v)
  end
  return w
end
# function cross{T}(v::Array{T,1},w::Array{T,2})
#   N=size(w,2)
#   @simd for idx=1:N
#     w[:,idx]=cross(w[:,idx],v)
#   end
#   return w
# end






function skew3{T}(v::Array{T,1})
  return [0 -v[3] v[2];v[3] 0 -v[1];-v[2] v[1] 0]
end

function integrate_Hp_midpoint{T}(self::pm_pif3d{T},x1n::Array{T,1},x2n::Array{T,1},
                       x3n::Array{T,1},
                       v1n::Array{T,1},v2n::Array{T,1}, v3n::Array{T,1},
                      qn::Array{T,1}, mn::Array{T,1},wn::Array{T,1},
          J1::Array{Complex{T},3},J2::Array{Complex{T},3},J3::Array{Complex{T},3},
          B1::Array{Complex{T},3},B2::Array{Complex{T},3}, B3::Array{Complex{T},3}, dt::T)
  Np::Int=length(x1n)

  psi1=Array{Complex{T}}(self.N1,self.N2,self.N3)
  psi0=Array{Complex{T}}(self.N1,self.N2,self.N3)

  psix1=Array{Complex{T}}(self.N1)
  psix2=Array{Complex{T}}(self.N2)
  psix3=Array{Complex{T}}(self.N3)

  #Fourier stem function Integral
  K1,K2,K3=getK(self)

  vol::T = 1. / (self.L1 * self.L2 * self.L3)

  Bn=Array(T,3)
  x05=Array(T,3)
  v=Array(T,3)
  v0=Array(T,3)
  TEMP=Array{Complex{T}}(3,3)
  DBn=Array(T,3,3)
  F=Array(T,3)
  DF=Array(T,3,3)

  for pdx=1:Np
    q_m::T=qn[pdx]/mn[pdx]

    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi0)

    #v0.=[v1n[pdx],v2n[pdx],v3n[pdx]]
    v0[1]=v1n[pdx]; v0[2]=v2n[pdx]; v0[3]=v3n[pdx]
    v[1]=v0[1]; v[2]=v0[2];v[3]=v0[3]
    #v.=v0
    error::T=Inf
    iter::Int=1
    while (error >1e-14 && iter<8)
      iter+=1
    #x05.=[x1n[pdx],x2n[pdx],x3n[pdx]]+dt/4*(v+v0)
    x05[1]=x1n[pdx]+dt/4*(v[1]+v0[1])
    x05[2]=x1n[pdx]+dt/4*(v[2]+v0[2])
    x05[3]=x1n[pdx]+dt/4*(v[3]+v0[3])
    eval_basis(self,x05...,psi1)
    Bn[1]=eval_scalar(self, psi1 ,B1)
    Bn[2]=eval_scalar(self, psi1 ,B2)
    Bn[3]=eval_scalar(self, psi1 ,B3)

    #Picard
    F.=v0 .+ (dt/2*q_m)*cross(v0.+v, Bn)
    error=maximum(abs.(v-F ))
    v.=F

    # eval_scalar_gradient(self, psi1 ,B1,TEMP[:,1])
    # eval_scalar_gradient(self, psi1 ,B2,TEMP[:,2])
    # eval_scalar_gradient(self, psi1 ,B3,TEMP[:,3])
    # DBn.=real(TEMP)'
    # #Newton
    # F.=v0 .+ (dt/2*q_m)*cross(v0.+v, Bn) -v
    # DF.=-eye(Complex{T},3) +
    #   (dt/2*q_m)*(-skew3(Bn)+ dt/4*cross!(v0.+v, DBn))
    #
    # F.=v - (DF\F)
    # error=maximum(abs(v-F ))
    # v.=F
    end

    # Push with new velocity
    v1n[pdx]=v[1];v2n[pdx]=v[2];v3n[pdx]=v[3]
    x1n[pdx]+=dt*(v[1]+v0[1])/2
    x2n[pdx]+=dt*(v[2]+v0[2])/2
    x3n[pdx]+=dt*(v[3]+v0[3])/2
    v[1]=(v[1]+v0[1])/2
    v[2]=(v[2]+v0[2])/2
    v[3]=(v[3]+v0[3])/2
    eval_basis(self,x1n[pdx],x2n[pdx],x3n[pdx],psi1)

    # B1n::T=eval_scalar(self, psi0 ,B1)
    # eval_scalar_gradient(self, psi0 ,B1,DB1n)
    # B2n::T=eval_scalar(self, psi0 ,B2)
    # eval_scalar_gradient(self, psi0 ,B2,DB2n)
    # B3n::T=eval_scalar(self, psi0 ,B3)
    # eval_scalar_gradient(self, psi0 ,B3,DB3n)

    #Accumulate charge with linear trajectory
    w::T=wn[pdx]*qn[pdx]*vol #Accumulated weight with fourier normalization

    conj!(psi1)
    conj!(psi0)
    for kdx=1:self.N3
      kv3::T=K3[kdx]*v[3]
      for jdx=1:self.N2
        kv23::T=K2[jdx]*v[2]+kv3
       @simd for idx=1:self.N1
        kv123::T=K1[idx]*v[1]+kv23
        psi123::Complex{T}=im*(psi1[idx,jdx,kdx]-psi0[idx,jdx,kdx])/kv123*w
        if kv123==0 # DIRTY
          psi123=dt*w
        end

       J1[idx,jdx,kdx]+=v[1]*psi123
       J2[idx,jdx,kdx]+=v[2]*psi123
       J3[idx,jdx,kdx]+=v[3]*psi123
       end
      end
    end
  end
end



#
#
# function integrate_Hp1{T}(self::pm_pif3d, d_E1::Array{Complex{T},3},
#           B2,B3
#            x1n::Array{T,1},x2n::Array{T,1}, x3n::Array{T,1},
#            v1n::Array{T,1},v2n::Array{T,1}, v3n::Array{T,1},wn::Array{T,1},
#            qn::Array{T,1}, mn::Array{T,1}, dt::T)
#
#
#            Np=length(x1n)
#            @assert Np==length(x2n)
#            @assert Np==length(x3n)
#
#            #Fn=zeros(Complex{T},Np)
#            Fn[:]=0
#            psi1=Array{Complex{T}}(self.N1)
#            psi1_=Array{Complex{T}}(self.N1)
#            psi2=Array{Complex{T}}(self.N2)
#            psi3=Array{Complex{T}}(self.N3)
#            k1=2*pi/self.L1
#            k2=2*pi/self.L2
#            k3=2*pi/self.L3
#            d_E1=Array{Complex{T}}(self.N1,self.N2,self.N3)
#
#
#
#            for pdx=1:Np
#              qm::T=qn[pdx]./mn[pdx]
#              B1n::Complex{T}=0
#              B2n::Complex{T}=0
#              B3n::Complex{T}=0
#
#
#              #x1_::T=x1n[pdx]+dt*v1n[pdx]
#              #eval_fourier_modes0(x1_*k1, self.Nx1,psi1_)
#
#              eval_fourier_modes0(x1n[pdx]*k1, self.Nx1,psi1)
#              eval_fourier_modes2(x2n[pdx]*k2, self.Nx2,psi2)
#              eval_fourier_modes2(x3n[pdx]*k3, self.Nx3,psi3)
#
#             #H_p1
#             for kdx=1:self.N3
#                 for jdx=1:self.N2
#                 psi23::Complex{T}=psi2[jdx]*psi3[kdx]
#
#
#
#                 vn2[pdx]+=(-qm)*dt*vn1[pdx]*real(psi23*B3[1,jdx,kdx])
#                 vn3[pdx]+=(qm)*dt*vn1[pdx]*real(psi23*B2[1,jdx,kdx])
#                 d_E1[1,jdx,kdx]-= dt*qn[pdx]*wn[pdx]*vn1[pdx]*psi23
#
#
#                  @simd  for idx=2:self.N1
#                     vn2[pdx]+=(-qm)*dt*vn1[pdx]*
#
#
#
#                    psin::Complex{T}=psi1[idx]*psi23
#                    psin_::Complex{T}=psi1_[idx]*psi23
#
#
#                     Fn[pdx]+=(real(psin)*real(F[idx,jdx,kdx])
#                              -imag(psin)*imag(F[idx,jdx,kdx]))
#                  end
#                end
#              end
#
#            end
#
#            d_E1=conj(d_E1)/self.L1/self.L2/self.L3 #Normalize
#
#
#
#
#
#
#           xk1=xk1+ (deltat)*vk1;
#           psik1=exp(1j*( xk1*KX1.'+xk2*KX2.'+xk3*KX3.' ));
#
#           vk2=vk2 + (-qk./mk).*(evE_mask( psik1-psik0, B3./(1j*KX1),KDX1 ) ...
#               + deltat*vk1.*evE_mask(psik1, B3,~KDX1));
#           vk3=vk3+ (qk./mk).*(evE_mask( psik1-psik0, B2./(1j*KX1),KDX1 )...
#               + deltat*vk1.*evE_mask(psik1, B2,~KDX1));
#
#           E1(KDX1)=E1(KDX1) + mean(bsxfun(@times, wk.*qk,...
#               conj(psik1(:,KDX1) -psik0(:,KDX1)) )).'./(1j.*KX1(KDX1))...
#               /L1/L2/L3;
#           E1(~KDX1)= E1(~KDX1) - (deltat)*mean(bsxfun(@times, ...
#               wk.*vk1.*qk, conj(psik1(:,~KDX1)))).'/L1/L2/L3;
#           E1(~KDX)=0;
#           psik0=psik1;
#
#
# end

# pm=pm_pif3d{Float64}(4,5,6,1.0)
# using BenchmarkTools
#
# psin=rand(pm.N1,pm.N2,pm.N3)+rand(pm.N1,pm.N2,pm.N3)*im
# F=rand(pm.N1,pm.N2,pm.N3)+rand(pm.N1,pm.N2,pm.N3)*im
# @benchmark val=eval_scalar(pm,psin,F)
#
# @time val=eval_scalar(pm,psin,F)
# @benchmark val=eval_scalar2(pm,psin,F)
# end



## MPI tools

#
#
#
#
#
# type pm_pif3d{T}
#   Nx1::Int
#   Nx2::Int
#   Nx3::Int
#   L1::T
#   L2::T
#   L3::T
#   N1::Int
#   N2::Int
#   N3::Int
#   # Unit mode
#   kx1::T
#   kx2::T
#   kx3::T
#   K1::Array{T}{1}
#   K2::Array{T}{1}
#   K3::Array{T}{1}
#   # Indicies of zero modes
#   N1n::Int
#   N2n::Int
#   N3n::Int
#
#   function pm_pif3d{T}(Nx1::Int,Nx2::Int,Nx3::Int,
#                                 L1::T,L2::T,L3::T) where T<:AbstractFloat
#     self=new(Nx1,Nx2,Nx3,L1,L2,L3,Nx1+1,Nx2*2+1,Nx3*2+1)
#
#     self.kx1=2*pi/self.L1
#     self.kx2=2*pi/self.L2
#     self.kx3=2*pi/self.L3
#     self.N1n=1
#     self.N2n=self.Nx2+1
#     self.N3n=self.Nx3+1
#     self.K1,self.K2,self.K3=getK(self)
#     return self
#   end
#   function pm_pif3d{T}(Nx1::Int,Nx2::Int,Nx3::Int,L::T) where T
#      pm_pif3d(Nx1,Nx2,Nx3,L,L,L)
#   end
# end
# # Outer constructor to infer type T directly from given length
# pm_pif3d(Nx1::Int,Nx2::Int,Nx3::Int,L1::T,L2::T,L3::T) where
#       {T}=pm_pif3d{T}(Nx1,Nx2,Nx3,L1::T,L2::T,L3::T)
# pm_pif3d(Nx1::Int,Nx2::Int,Nx3::Int,L::T) where
#             {T<:AbstractFloat}=pm_pif3d{T}(Nx1,Nx2,Nx3,L)
#


# using Cubature
# Project to grid
function to_grid(self::pm_pif3d, scalar_fun)

    if (scalar_fun==nothing)
        return zeros(Complex{Float64}, self.N1,self.N2,self.N3)
    end

    sf=(4/3)*2;
    N::Array{Int,1}=nextpow.( 2, 32+[self.Nx1+1;self.Nx2+1;self.Nx3+1].*sf);
    L=[self.L1;self.L2;self.L3]
    dx=L./N

    x1=reshape(collect(0:N[1]-1)*dx[1],N[1],1,1 )
    x2=reshape(collect(0:N[2]-1)*dx[2],1,N[2],1 )
    x3=reshape(collect(0:N[3]-1)*dx[3],1,1,N[3] )

    coefs=fft(scalar_fun(x1,x2,x3) .+ 0. * x1 .+ 0. * x2 .+ 0. * x3,1:3)[
                   (1:1:self.Nx1+1),
            [ N[2]-(self.Nx2-1):1:N[2];1:1:self.Nx2+1 ],
            [ N[3]-(self.Nx3-1):1:N[3];1:1:self.Nx3+1 ] ]/prod(N);

    return coefs
end
