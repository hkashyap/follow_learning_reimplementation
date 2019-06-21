# plots tuning curve of a LIF neuron
#%%
import numpy as np
import matplotlib.pyplot as plt
#import ipdb; ipdb.set_trace()

gain = 1
bias = 0
vReset = 0
vTh = 1
tau_m = 20#0.01
tau_ref = 2
vmin = 0

def lif_neuron(v, refractory, i):
    # refractory period
    if refractory > 0:
        return (vReset, refractory-1, 0)
        
    dvdt = (-v + i)/tau_m
    #v = max(v + dvdt, vmin)
    v = v + dvdt

    if v >= vTh:
        return (vReset, tau_ref, 1)
    else:
        return (v, 0, 0)    

def createUniformGroup(nNeuron):
    gain = np.ones(nNeuron)
    bias = np.zeros(nNeuron)
    # find varied gain and biases for L1 neurons
    for n in range(nNeuron):
        expRate = (200 + (300/nNeuron)*n)/1000
        gain[n] = ((vReset-vTh) * np.exp(1/(tau_m*expRate))) /(1 - np.exp(1/(tau_m*expRate)))
    return (gain, bias)

def createHeteroGroup(nNeuron):
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

if __name__=="__main__":
    # plot the response of the neuron for a constant input 
    # simTime = 40
    # I = 1.5

    # V = np.zeros(simTime)
    # spikes = np.zeros(simTime)
    # refractory = 0
    # for t in range(1,40,1):
    #     (V[t], refractory, spikes[t]) = lif_neuron(V[t-1], refractory, gain*I+bias)
    # #plt.plot(V/vTh)
    # #plt.plot(np.ones(simTime))
    # #plt.show()

    # # plot tuning curve of the LIF neuron (has a particular gain and bias)
    # # by running the dynamics
    # fr_dynamics = []
    # isteps = np.linspace(0,8,num=100)
    
    # for I in isteps:
    #     V = 0
    #     spikes = 0
    #     refractory = 0
    #     for t in range(1,1000,1):
    #         (V, refractory, spike) = lif_neuron(V, refractory, gain*I+bias)
    #         spikes = spikes + spike
    #     fr_dynamics.append(spikes)
    
    # # by analytical form
    # fr_analytical = []
    # for I in isteps:
    #     j = gain*I+bias
    #     fr_analytical.append(1000/(tau_ref + (tau_m*np.log(j/(j-vTh)))) if j>vTh else 0)

    # #plt.plot(isteps, fr_dynamics)
    # #plt.plot(isteps, fr_analytical)
    # #plt.show()

    # createUniformGroup - create an ensemble of LIF neurons
    nNeuron = 50
    isteps = np.linspace(-2,5,num=100)
    fr_ensemble = np.zeros(shape = (nNeuron, len(isteps)))
    fr_ensemble_a = np.zeros(shape = (nNeuron, len(isteps)))
    j_ensemble = np.zeros(shape = (nNeuron, len(isteps)))

    # (gain, bias) = createUniformGroup(nNeuron)
    # for n in range(nNeuron):
    #     i = 0
    #     for I in isteps:
    #         j = gain[n]*I+bias[n]
    #         fr_ensemble[n,i] = 1000/(tau_ref + (tau_m*np.log(j/(j-vTh)))) if j>vTh else 0
    #         i+=1
    # fr_ensemble[np.isnan(fr_ensemble)] = 0
    # plt.plot(isteps, np.transpose(fr_ensemble))
    # plt.show()

    # createHeteroGroup
    (gain, bias) = createHeteroGroup(nNeuron)
    for n in range(nNeuron):
        i = 0
        for I in isteps:
            j = gain[n]*I+bias[n]
            V = 0
            spikes = 0
            refractory = 0
            for t in range(1,1000,1):
                (V, refractory, spike) = lif_neuron(V, refractory, j)
                spikes = spikes + spike
            fr_ensemble[n, i] = spikes
            fr_ensemble_a[n,i] = 1000/(tau_ref + (tau_m*np.log(j/(j-vTh)))) if j>vTh else 0
            j_ensemble[n,i] = j
            i+=1
            
    plt.plot(isteps, np.transpose(j_ensemble))
    plt.title("Input current")
    plt.show()

    fr_ensemble[np.isnan(fr_ensemble)] = 0
    plt.plot(isteps, np.transpose(fr_ensemble))
    plt.title("Tuning curves of real LIF neurons")
    plt.show()
    
    fr_ensemble_a[np.isnan(fr_ensemble_a)] = 0
    plt.plot(isteps, np.transpose(fr_ensemble_a))
    plt.title("analytical tuning curves")
    plt.show()

#print('gain:',gain)
#print('bias:',bias)

#plt.plot(isteps, np.transpose(fr_ensemble[1,:]))
#plt.show()
