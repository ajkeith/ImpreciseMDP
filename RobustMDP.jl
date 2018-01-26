########################################################
#
# Robust MDP
#
# Andrew Keith
# 24 Jan 2018
# UNCLASSIFIED, Distribution Unlimited (A)
# MIT license
#
########################################################

############################
# Define constants
###########################

# Epochs
# finite horizon
N = 5

# States
S = [1,2,3]

# Actions
A = [1,2,3]

# Rewards
R = [1.0 10 10; 10 10 10; 2 2 2]

rN = [1; 10.0; 2.0]

# Uncertainty Sets
P = zeros(Float64,3,3,3)
P[:,:,1] = [0.8 0.1 0.1; 0.8 0.1 0.1; 0.8 0.1 0.1]
P[:,:,2] = [0.1 0.8 0.1; 0.1 0.8 0.1; 0.1 0.8 0.1]
P[:,:,3] = [0.1 0.1 0.8; 0.1 0.1 0.8; 0.1 0.1 0.8]


# Bisection Algorithm - Entropy Uncertainty Sets

function sig(lam::Float64,beta::Float64, v::Array{Float64,1}, q::Array{Float64,1})
  lam * log(q' * exp.(v / lam)) + beta * lam
end

function biEntropy(del::Float64, beta::Float64, vimax::Float64, v::Array{Float64,1}, q::Array{Float64,1})
  result = 0.0
  if beta == 0
    result = q' * v
  else
    laml = 0 + del
    lamu = (vimax - q' * v) / beta
    lam = (laml + lamu) / 2

    while lamu - laml > del
      lam = (laml + lamu) / 2
      if sig(lam,beta,v,q) < sig(laml,beta,v,q)
        laml = copy(lam)
      else
        lamu = copy(lam)
      end
    end
    result = sig(lam,beta,v,q)
  end
  result
end

q = P[1,:,1]
v = rN
vimax = maximum(v)

tempsigs = Vector{Float64}(1)

nS = size(S,1)
nA = size(A,1)
eps = 1e-5
del = eps / N
policy = zeros(Int64,nS,N -1)
value = zeros(Float64,nS,N)
value[:,N] = rN
beta = 12.0

for t = (N-1):-1:1
  v = value[:,t + 1]
  vimax = maximum(v)
  for i = 1:nS
    vmin = Inf
    amin = 0
    for j = 1:nA
      sighat = biEntropy(del,beta,vimax,v,P[i,:,j])
      # println("i = $i, j = $j, vmin = $vmin, sighat = $sighat")
      if (R[i,j] + sighat) < vmin
        vmin = R[i,j] + sighat
        amin = j
      end
    end
    value[i,t] = vmin
    policy[i,t] = amin
  end
end

println(policy)
println(value)
