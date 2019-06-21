# module to check LIF neurons
#%%
import numpy as np
import matplotlib.pyplot as plt

# simulation
simTime = 10
nSyn = 100

# generate random spikes - 1 for spike
spike_train = np.random.random_integers(0, 1, size=(nSyn,simTime))

# uncomment to generate a single spike
# nSyn = 1
# spike_train = np.zeros([nSyn,20])
# spike_train = np.append(spike_train, np.ones([nSyn,1]),axis=1)
# spike_train = np.append(spike_train, np.zeros([nSyn,29]),axis=1)

# neuron parameters
vTh = 1
vReset = 0
tau_s = 20
tau_m = 20
refractoryPeriod = 2

# kernel for spike integration
k = [np.exp(-t/tau_s)/tau_s for t in range(tau_s)]
#plt.plot(k)
#plt.show()

# states
currentV = vReset
v = np.zeros(simTime)
I = np.zeros(simTime)
outSpikes = np.zeros(simTime)
currentRefractory = 0

# run simulation
for t in range(simTime):
    #find total input current to the neuron
    for syn in range(nSyn):
        spikes = np.append(np.zeros(max(0, tau_s - t -1)), spike_train[syn, max(0, t+1-tau_s):t+1])
        I[t] =  I[t] + np.convolve(spikes, k, 'valid')

    # refractory
    if currentRefractory > 0:
        currentRefractory = currentRefractory-1
        continue

    dvdt = (-currentV + I[t])/tau_m
    v[t] = currentV = currentV + dvdt
    if currentV >= 1:
        outSpikes[t] = 1
        v[t] = currentV = vReset
        currentRefractory = refractoryPeriod

    # #print(len(np.convolve(spike_train[max(0, t+1-tau):t+1], k, 'valid')))

#print(len(spike_train[max(0, t+1-tau):t+1]), len(k))
#print(k)

#np.convolve(spike_train[0:31], k, 'valid')

#plt.plot(spike_train)
#plt.plot(I)
plt.plot(v)
plt.plot(outSpikes)
plt.show()
print('total input spikes: ', np.sum(spike_train))
