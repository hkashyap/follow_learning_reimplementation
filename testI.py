#%%
from matplotlib import pyplot as plt
import numpy as np
#from netConfig import *

nNeuronsL1 = 100
refractoryPeriod = 2
tau_m = 20
vTh = 1
vReset = 0

def lif_neuron(v, refractory, i):
        # refractory period
        if refractory > 0:
            return (vReset, refractory-1, 0)

        if v >= vTh:
            return (vReset, refractoryPeriod, 1)
        else:
            dvdt = (-v + i)/tau_m
            v = max(v + dvdt, 0)
            return (v, 0, 0)

# neuron specifc biases and gains
#biasL1 = np.zeros(nNeuronsL1)
#gainL1 = np.ones(nNeuronsL1)
rmL1 = np.ones(nNeuronsL1)

# find bias and gain for L1 neurons
for n in range(nNeuronsL1):
    expRate = (200 + (300/nNeuronsL1)*n)/1000
    #gainL1[n] = np.random.randint(1,10)
    #gainL1[n] = 1
    #biasL1[n] = -gainL1[n] + 1/(1-np.exp(((1/expRate)-refractoryPeriod)/tau_m))
    rmL1[n] = ((vReset-vTh) * np.exp(1/(tau_m*expRate))) /(1 - np.exp(1/(tau_m*expRate)))

for n in range(nNeuronsL1):
    #outrate = 1/(refractoryPeriod+20*np.log(1 - (1/(gainL1[n]+biasL1[n]))))
    outrate = 1/(tau_m*np.log((rmL1[n])/(rmL1[n] + vReset - vTh)))
    print(n, ':: rm:', rmL1[n], 'e:', 200 + (300/nNeuronsL1)*n, 'rate:', outrate)

rtest = np.zeros(100)
vtest = np.zeros(100)
spikesTotal = np.zeros(100)
for n in range(100):
    for timeTest in range(10000):
        (vtest[n], rtest[n], spikeTest) = lif_neuron(vtest[n], rtest[n], rmL1[n])
        spikesTotal[n] = spikesTotal[n] + spikeTest

plt.plot(rmL1, spikesTotal/10)
plt.show()