import SpecialFunctions: erfinv
using Sobol


function uniform_6d(Np::Int, nskip = 0, T = Float64)

    x1n = Array{T}(Np)
    x2n = Array{T}(Np)
    x3n = Array{T}(Np)
    v1n = Array{T}(Np)
    v2n = Array{T}(Np)
    v3n = Array{T}(Np)

    sob = Sobol.SobolSeq(6)
    Sobol.skip(sob, 4 + nskip) # Skip some entries
    for i = 1:Np
        x1n[i], x2n[i], x3n[i], v1n[i], v2n[i], v3n[i] = Sobol.next(sob)
    end

    x1n, x2n, x3n, v1n, v2n, v3n

end

# Sampling

function maxwellian_6d(
    x1n,
    x2n,
    x3n,
    v1n,
    v2n,
    v3n,
    L::Array{T,1} = ones(T, 3),
    sigma::Array{T,1} = ones(T, 3),
    mu::Array{T,1} = zeros(T, 3),
) where {T}

    x1n .*= L[1]
    x2n .*= L[2]
    x3n .*= L[3]

    norminv = p -> sqrt(2.0) .* erfinv.(2.0 * p - 1.0)

    v1n .= norminv(v1n) .* sigma[1] .+ mu[1]
    v2n .= norminv(v2n) .* sigma[2] .+ mu[2]
    v3n .= norminv(v3n) .* sigma[3] .+ mu[3]

    #Probability density
    p(x1, x2, x3, v1, v2, v3) =
        exp.(-(v1 .- mu[1]) .^ 2 ./ sigma[1]^2 / 2.0 .-
             (v2 .- mu[2]) .^ 2 ./ sigma[2]^2 / 2.0 .-
             (v3 .- mu[3]) .^ 2 ./ sigma[3]^2 / 2.0) .* (2 * pi)^(-1.5) /
        (prod(sigma) * prod(L))


    p

end

function maxwellian_6d(
    Np::Int,
    nskip::Int = 0,
    L::Array{T,1} = ones(T, 3),
    sigma::Array{T,1} = ones(T, 3),
    mu::Array{T,1} = zeros(T, 3),
) where {T}

    x1n, x2n, x3n, v1n, v2n, v3n = uniform_6d(Np, nskip)

    p = maxwellian_6d(x1n, x2n, x3n, v1n, v2n, v3n, L, sigma, mu)

    x1n, x2n, x3n, v1n, v2n, v3n, p

end

# Rotation of initial density for testing
# function rotate3d
#
# end

# input uniformly distributed
function maxwellian_stream_3d(
    v1n,
    v2n,
    v3n,
    alpha::Array{T,1},
    sigma::Array{T,2},
    mu::Array{T,2},
) where {T}
    Ns::Int = length(alpha) # number of streams
    Np::Int = length(v1n)
    norminv = p -> sqrt(2.0) .* erfinv.(2.0 * p - 1.0)
    v1n .= norminv(v1n)
    v2n .= norminv(v2n)
    v3n .= norminv(v3n)

    nidx = Array{Int}(Ns + 1)
    nidx .= [1; ceil(Int, alpha[1:Ns-1] * Np); Np]

    for ndx = 1:Ns
    # use fused multiply add fma
        v1n[nidx[ndx]:nidx[ndx+1]] .= fma.(
            v1n[nidx[ndx]:nidx[ndx+1]],
            sigma[1, ndx],
            mu[1, ndx],
        )
        v2n[nidx[ndx]:nidx[ndx+1]] .= fma.(
            v2n[nidx[ndx]:nidx[ndx+1]],
            sigma[2, ndx],
            mu[2, ndx],
        )
        v3n[nidx[ndx]:nidx[ndx+1]] .= fma.(
            v3n[nidx[ndx]:nidx[ndx+1]],
            sigma[3, ndx],
            mu[3, ndx],
        )
    end

    function p(v1, v2, v3)

        fun = similar(v1)
        fun .= 0
        for ndx = 1:Ns
            fun += exp.(-(v1 - mu[1, ndx]) .^ 2 ./ sigma[1, ndx]^2 / 2.0 -
                        (v2 - mu[2, ndx]) .^ 2 ./ sigma[2, ndx]^2 / 2.0 -
                        (v3 - mu[3, ndx]) .^ 2 ./ sigma[3, ndx]^2 / 2.0) *
                   (2 * pi)^(-1.5) / prod(sigma[:, ndx])
        end
        return fun
    end

    return p
end

d = Dict("a" => 1, "b" => 2, "c" => 3);


"""
Linear Landau Damping (Langmuir Wave)
# Arguments
- epsilon: default 0.5 for strong landau damping, set 0.05 for weak (linear)
"""
function landau(
    x1n::Array{T,1},
    x2n,
    x3n,
    v1n,
    v2n,
    v3n;
    epsilon = 0.5,
    sigma = ones(T, 3),
    k::Array{T,1} = [0.5, 0.5, 0.5],
    q = -1.0,
    m = 1.0,
) where {T}

    L = ones(T, 3) * 2 * pi ./ k
    g = maxwellian_6d(x1n, x2n, x3n, v1n, v2n, v3n, L, sigma, zeros(T, 3))

    f(x1, x2, x3, v1, v2, v3) =
        (1.0 + epsilon * cos.(k[1] * x1)) .* g(x1, x2, x3, v1, v2, v3) * prod(L)

    # Default parameters
    params = Dict(
        "dt" => 0.1,
        "tmax" => 25,
        "q" => q,
        "m" => m,
        "c" => 1.0,
        "sigma" => sigma,
        "k" => k,
        "epsilon" => epsilon,
        "L" => L,
        "f0" => f,
        "g0" => g,
    )
    return params
end

"""
Jeans instability
"""
function jeans(
    x1n::Array{T,1},
    x2n,
    x3n,
    v1n,
    v2n,
    v3n;
    epsilon = 0.1,
    sigma = ones(T, 3),
    k::Array{T,1} = [0.5, 0.5, 0.5],
    q = -1.0,
    m = -1.0,
) where {T}
    @assert m < 0
    landau(
        x1n,
        x2n,
        x3n,
        v1n,
        v2n,
        v3n,
        epsilon = epsilon,
        sigma = sigma,
        k = k,
        q = q,
        m = m,
    )
end



"Weibel instability in six dimensions"
function weibel(
    x1n,
    x2n,
    x3n,
    v1n,
    v2n,
    v3n;
    epsilon = 1e-3,
    beta = -1e-3,
    sigma = [0.02 / sqrt(2), 0.02 * sqrt(12 / 2), 0.02 * sqrt(12 / 2)],
    k::Array{T,1} = [1.25, 1.25, 1.25],
) where {T}

    L = ones(T, 3) * 2 * pi ./ k

    g = maxwellian_6d(x1n, x2n, x3n, v1n, v2n, v3n, L, sigma, zeros(T, 3))

    f(x1, x2, x3, v1, v2, v3) =
        (1.0 + epsilon * sin.(k[1] * x1)) .* g(x1, x2, x3, v1, v2, v3) * prod(L)

    B3(x1, x2, x3) = real(beta) * cos.(k[1] * x1) + imag(beta) * sin.(k[1] * x1)

  # Default parameters
    params = Dict(
        "dt" => 0.2,
        "tmax" => 150,
        "q" => -1.0,
        "m" => 1.0,
        "c" => 1.0,
        "sigma" => sigma,
        "k" => k,
        "epsilon" => epsilon,
        "L" => L,
        "B" => (nothing, nothing, B3),
        "f0" => f,
        "g0" => g,
    )

    return params
end

export weibel_streaming


"Weibel streaming instability in six dimensions"
function weibel_streaming(
    x1n::Array{T,1},
    x2n,
    x3n,
    v1n,
    v2n,
    v3n;
    epsilon::T = T(0),
    beta = -im * 1e-3,
    sigma::Array{T,1} = 0.1 ./ sqrt(2) .* ones(T, 3),
    k::Array{T,1} = [0.2; 0.2; 0.2],
    delta::T = (1.0 / 6.0),
) where {T}
    L = ones(T, 3) * 2 * pi ./ k
    x1n .*= L[1]
    x2n .*= L[2]
    x3n .*= L[3]

    v01 = 0.5
    v02 = -0.1


    gv = maxwellian_stream_3d(
        v1n,
        v2n,
        v3n,
        [delta; 1.0 - delta],
        sigma .* ones(1, 2),
        [[0.0 0.0]; [v01 v02]; [0.0 0.0]],
    )

    g(x1, x2, x3, v1, v2, v3) =
        (1.0 .+ epsilon * sin.(k[1] * x1)) .* gv(v1, v2, v3) ./ prod(L)
    f(x1, x2, x3, v1, v2, v3) =
        (1.0 .+ epsilon * sin.(k[1] * x1)) .* gv(v1, v2, v3)

    B3(x1, x2, x3) = real(beta) * cos.(k[1] * x1) + imag(beta) * sin.(k[1] * x1)

  # Default parameters
    params = Dict(
        "dt" => 0.1,
        "tmax" => 200,
        "q" => -1.0,
        "m" => 1.0,
        "c" => 1.0,
        "sigma" => sigma,
        "k" => k,
        "epsilon" => epsilon,
        "L" => L,
        "B" => (nothing, nothing, B3),
        "f0" => f,
        "g0" => g,
    )

    return params
end

"""
Two dimensional periodic Kelvin Helmholtz instability

# Arguments
 k - k=1 is the stable mode in periodic domain
     chose k<1 for instability in corresponding dimension
     domain stability in 2-3
"""
function KelvinHelmholtz(
    x1n::Array{T,1},
    x2n,
    x3n,
    v1n,
    v2n,
    v3n;
    epsilon::T = 0.015,
    k::Array{T,1} = [0.4; 1.0; 1.0],
    q = -1.0,
    m = 1.0,
    B0::Array{T,1} = [0.0; 0.0; 10.0],
) where {T}

    L = ones(T, 3) * 2 * pi ./ k

    g = maxwellian_6d(x1n, x2n, x3n, v1n, v2n, v3n, L)

    f(x1, x2, x3, v1, v2, v3) =
        (1 + sin.(k[2] .* x2) + epsilon .* cos.(k[1] .* x1)) .*
        exp.(-0.5 .* (v1 .^ 2.0 + v2 .^ 2.0 + v3 .^ 2)) ./ sqrt(2 * pi) .^ 3

# Default parameters
    params = Dict(
        "dt" => 0.1 / norm(B0),
        "tmax" => 20 * norm(B0),
        "q" => q,
        "m" => m,
        "c" => 1.0,
        "k" => k,
        "epsilon" => epsilon,
        "L" => L,
        "B" => (
            (x1, x2, x3) -> B0[1],
            (x1, x2, x3) -> B0[2],
            (x1, x2, x3) -> B0[3],
        ),
        "f0" => f,
        "g0" => g,
    )



    return params

end
