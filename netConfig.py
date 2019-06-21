# this modeule defines parameters and functions for 
# network configuration and updates

#%%
# imports
import numpy as np
from netParam import *
import matplotlib.pyplot as plt

# kernel for spike integration
k = np.array([np.exp(-t/tau_s)/tau_s for t in range(tau_s)]).reshape(1, tau_s)
k = np.fliplr(k)

kLearn = np.array([np.exp(-t/tau_learn)/tau_learn for t in range(tau_learn)]).reshape(1, tau_learn)
kLearn = np.fliplr(kLearn)

vL1 = np.zeros(nNeuronsL1)
refractoryL1 = np.zeros(nNeuronsL1)
vL2 = np.zeros(nNeuronsL2)
refractoryL2 = np.zeros(nNeuronsL2)

# synapse weights: pre - post
wSynCmdL1 = np.random.randn(dimCommand, nNeuronsL1) * 0.3# mu =1 sigma = 0.3
wFFsynL2 = np.random.randn(nNeuronsL1, nNeuronsL2) * 0#0.001 / nNeuronsL1 #+ 0.01
#wFFsynL2 = np.zeros([nNeuronsL1, nNeuronsL2])
#wRsynL2 = np.random.randn(nNeuronsL2, nNeuronsL2) * 0.3
#wSynOut = np.random.randn(nNeuronsL2, dimOutput) * 0.3
# wFFsynL2 = np.zeros([nNeuronsL1, nNeuronsL2])
wRsynL2 = np.zeros([nNeuronsL2, nNeuronsL2])
wSynOut = np.zeros([nNeuronsL2, dimOutput])
wErrFeed = np.random.randn(dimOutput, nNeuronsL2) * 0.3
sFeed = np.ones([dimOutput, nNeuronsL2])
#sFeed[:,0:round(np.shape(sFeed)[1]/2)]=-1

def lif_neuron(v, refractory, i):
    # refractory period
    #if refractory > 0:
    #    return (vReset, refractory-1, 0)
    
    v = v + ((-v + i) / tau_m)
    v = v * (refractory == 0)
    v = v * (v>vmin) + vmin * (v<=vmin)

    refractory = (refractory-1)*((refractory-1)>0)

    spikes = (v >= vTh)
    v = v * ~spikes
    refractory = tau_ref * spikes + refractory * ~spikes
    #dvdt = (-v + i)/tau_m
    #v = max(v + dvdt, vmin)
    #v = v + dvdt
    return v, refractory, spikes
    # if v >= vTh:
    #     return (vReset, tau_ref, 1)
    # else:
    #     return (v, 0, 0)


def createHeteroGroup(nNeuron):
    print("Heterogeneous LIF group created...")
    max_rates = np.linspace(.1, .3, nNeuron) # in kHz & ms
    np.random.shuffle(max_rates)
    intercepts = np.linspace(0, 0.8, nNeuron)

    """Analytically determine gain, bias."""
    inv_tau_ref = 1 / tau_ref if tau_ref > 0 else np.inf
    if np.any(max_rates > inv_tau_ref):
        raise ValueError("Max rates must be below the inverse "
                             "refractory period (%0.3f)")

    x = 1.0 / (1 - np.exp((tau_ref - (1.0 / max_rates)) / tau_m))
    gain = (1 - x) / (intercepts - 1.0)
    bias = 1 - gain * intercepts
    return (gain, bias)

# neuron specifc total membrane potential (Rm)
#rmL1 = np.zeros(nNeuronsL1)
#rmL2 = np.ones(nNeuronsL2)
(gainL1, biasL1) = createHeteroGroup(nNeuronsL1)
(gainL2, biasL2) = createHeteroGroup(nNeuronsL2)

# isteps = np.linspace(-1,2,num=100)
# fr_ensemble_a = np.zeros(shape = (nNeuronsL2, len(isteps)))
# for n in range(nNeuronsL2):
#     i = 0
#     for I in isteps:
#         j = gainL2[n]*I+biasL2[n]
#         fr_ensemble_a[n,i] = 1000/(tau_ref + (tau_m*np.log(j/(j-vTh)))) if j>vTh else 0
#         i+=1
# fr_ensemble_a[np.isnan(fr_ensemble_a)] = 0
# plt.plot(isteps, np.transpose(fr_ensemble_a))
# plt.show()

# find varied Rm for L1 neurons
#for n in range(nNeuronsL1):
#    expRate = (200 + (300/nNeuronsL1)*n)/1000
#    rmL1[n] = ((vReset-vTh) * np.exp(1/(tau_m*expRate))) /(1 - np.exp(1/(tau_m*expRate)))

# to test outrate
#for n in range(nNeuronsL1):
#    outrate = 1/(tau_m*np.log((rmL1[n])/(rmL1[n] + vReset - vTh)))

# find varied Rm for L2 neurons
#for n in range(nNeuronsL2):
#    expRate = (200 + (300/nNeuronsL2)*n)/1000
#    rmL2[n] = ((vReset-vTh) * np.exp(1/(tau_m*expRate))) /(1 - np.exp(1/(tau_m*expRate)))
