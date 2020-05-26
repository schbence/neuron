from pylab import *
from tqdm import tqdm

# System params
N = 500
T = 2000 #5000
ENS = 50 #10
PROC = 4

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

def evolve_neurons(w):
    seed()
    W = random([N,N])*(1-identity(N))*w

    Vs = zeros([T,N])
    Xs = zeros([T,N],dtype=bool)

    Xs[0] = newXs(0.5)

    for t in range(1,T):
        #Vs[t] = mu * Vs[t-1] + W/N * sum(Xs[t-1])
        Vs[t] = mu * Vs[t-1] + matmul(W,Xs[t-1])/N
        probs = phi(Vs[t], gamma)
        Xs[t] = newXs(probs) * bitwise_not(Xs[t-1])

    return Xs.mean(axis=1),Vs.mean(axis=1)

def multi_evolve(w,N,proc_N):
    from multiprocessing import Pool
    w_ens = [w] * N
    p = Pool(proc_N)
    return array(p.map(evolve_neurons, w_ens))

xx,vv = [],[]

Ws = linspace(1.05,1.125,10)
Ws = linspace(0.95,1.2,20)
nlast = 20

for w in tqdm(Ws):
    xsvs = multi_evolve(w, ENS, PROC)
    xx.append(xsvs[:,0,:])
    vv.append(xsvs[:,1,:])

xx = array(xx)
vv = array(vv)

name = 'parall_phase_trans_w_'+str(Ws[0])+'_'+str(Ws[-1])
save(name, array([xx,vv]))

xlast = xx[:,:,-nlast:].mean(axis=-1)

grid()

for i,w in enumerate(Ws):
    plot(repeat(w,xlast.shape[1]), xlast[i] , 'k.', alpha=0.5)

plot(Ws,xlast.mean(axis=1),'r-',alpha=0.8)



xlabel("$W$")
ylabel("$\\rho$")
savefig(name+'.png')





