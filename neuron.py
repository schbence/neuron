from pylab import *
from tqdm import tqdm

# System params
N = 500
T = 5000
W_ENS = 100

# Control params
gamma = 1.
mu    = 0.5

'''
def phi(v, gamma, vt, r):
    return (gamma*(v-vt))**r
'''

def phi(v, gamma):
    return gamma*v


def newXs(ps):
    return rand(N) < ps

# Variables
def evolve_neurons(w):
    W = random([N,N])*(1-identity(N))*w

    Vs = zeros([T,N])
    Xs = zeros([T,N],dtype=bool)

    Xs[0] = newXs(0.5)

    for t in range(1,T):
        #Vs[t] = mu * Vs[t-1] + W/N * sum(Xs[t-1])
        Vs[t] = mu * Vs[t-1] + matmul(W,Xs[t-1])/N
        probs = phi(Vs[t], gamma)
        Xs[t] = newXs(probs) * bitwise_not(Xs[t-1])

    return Xs,Vs

xx,vv = [],[]
Ws = linspace(1.02,1.2,100)
nshow = 20

for w in tqdm(Ws):
    xs,vs = evolve_neurons(w)
    xmean = xs.mean(axis=1)
    vmean = vs.mean(axis=1)
    xx.append(xmean)
    vv.append(vmean)

xx = array(xx)
vv = array(vv)

xm = xx[:,-nshow:].mean(axis=1)

name = 'phase_trans_w_'+str(Ws[0])+'_'+str(Ws[-1])

save(name, array([Ws,xm]))

grid()
plot(Ws, xm)
xlabel("$W$")
ylabel("$\\rho$")
savefig(name+'.png')





