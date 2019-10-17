





function Bpenning{T}( x1::T,x2::T,x3::T )
  B=zeros(T,3)
  JB=zeros(T,3,3)

  B[1]=1

  return B,JB
end

x1=1.;x2=2.;x3=3.;




function skew{T}(v::Array{T,1} )
  return [0  -v[3]  v[2]; v[3] 0 -v[1]; -v[2]  v[1]  0 ]
end





function odeHB{T}( x1::T,x2::T,x3::T,v1::T,v2::T,v3::T, q::T, m::T )

  B,JB=Bpenning(x1,x2,x3)
  F=[v1; v2; v3; cross([v1; v2; v3],B) ]
  DF=[zeros(T,3,3)  eye(T,3,3);skew([v1; v2; v3])*JB   -skew(B)]

 return F,DF
end

F,DF=odeHB(1.0,2.,3.,4.,5.,6.)


expm(DF*dt)*[x1;x2;x3;v1;v2;v3]



pinv(DF)*expm(DF)


deltat=0.1









cross(rand(3),rand(3,3))
