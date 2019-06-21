# This modeule runs the network in learning and
# testing phases, handles input and output

import matplotlib.pyplot as plt
from netConfig import *

# simulation parameters
tSim = 10000
tLearn = 9700
nMonN = 100
nMonW = 100

# target characterstics
freq = 1
peak = 6
eta = 0.003

K = -200

# generate random spikes - 1 for spike
#spike_train_ff = np.random.random_integers(0, 1, size=(nSynFF,tSim))

# generate command signal of dimension dimCommand
# command = np.append(np.ones(int(tSim/2)) * np.random.random(), np.ones(int(tSim/2)) * np.random.random())
command = np.append(np.ones(int(tSim/2)) * 0.9, np.ones(int(tSim/2)) * 0.9)

# to record output -  now obsolete
vt1 = np.zeros(tSim)
it1 = np.zeros(tSim)
vt2 = np.zeros(tSim)
it2 = np.zeros(tSim)

#it2ffr = np.zeros([nNeuronsL2, tSim])
#it2error = np.zeros([nNeuronsL2, tSim])

# spike tables
spikesL1 = np.zeros([nNeuronsL1, tau_s])
spikesL2 = np.zeros([nNeuronsL2, tau_s])

# spike monitor
spikeMonL1 = np.zeros([nMonN, tSim])
spikeMonL2 = np.zeros([nMonN, tSim])
monNL1 = np.floor(np.random.rand(nMonN)*nNeuronsL1).astype(int)
monNL2 = np.floor(np.random.rand(nMonN)*nNeuronsL2).astype(int)

# input monitor
iMonFFR = np.zeros(tSim)
iMonErr = np.zeros(tSim)

# weight monitor
wtMonRec = np.zeros([nMonW, tSim])
wtMonOut = np.zeros([nMonW, tSim])
monSrcN = np.floor(np.random.rand(nMonW)*nNeuronsL2).astype(int)
monDstN = np.floor(np.random.rand(nMonW)*nNeuronsL2).astype(int)

# output
output = np.zeros([dimOutput, tSim])
error = np.zeros([dimOutput, tSim])

# target
target = np.zeros([dimOutput, tSim])
for dim in range(dimOutput):
    for t in range(tSim):
        target[dim, t] = np.sin(np.pi/2 + (t*freq/1000)*2*np.pi)*peak
#target = np.ones([dimOutput, tSim]) * -3


if __name__=='__main__':

    # run the learning phase
    for t in range(tSim):
        # L1 neurons
        iL1 = command[t] * wSynCmdL1[0, :]

        # L2 neurons
        iL2 = np.zeros(nNeuronsL2) # intialize input current to L2 to zero
        #find total feed-forward current to the neuron
        #spikes = np.append(np.zeros([nNeuronsL1, max(0, tau_s - t)]), spikesL1[:, max(0, t-tau_s):t], axis=1)        
        #spikes = spikesL1
        iL2 = iL2 + np.dot(np.dot(k, np.transpose(spikesL1)), wFFsynL2)

        #find total recurrent current to the neuron
        #spikesR = np.append(np.zeros([nNeuronsL2, max(0, tau_s - t)]), spikesL2[:, max(0, t-tau_s):t], axis=1)
        spikesR = spikesL2
        iL2 = iL2 + np.dot(np.dot(k, np.transpose(spikesR)), wRsynL2)
        
        #it2ffr[:, t] = iL2 # record feed-forward + recurrent current to L2 neuron
        iMonFFR[t] = np.mean(iL2)

        if t<=tLearn:
            errorTraceInt = np.append(np.zeros([dimOutput, max(0, tau_s - t)]), error[:, max(0, t-tau_s):t], axis=1)
            #Error feedback
            #iL2 = iL2 + np.dot(k, np.transpose(errorTraceInt)) * wErrFeed * K
            iL2 = iL2 + np.dot(k, np.transpose(errorTraceInt)) * sFeed * K

            # record error feedback current to L2 neuron
            #iMonErr[t] = np.mean(np.dot(k, np.transpose(errorTraceInt)) * wErrFeed * K)
            iMonErr[t] = np.mean(np.dot(k, np.transpose(errorTraceInt)) * sFeed * K)

            #it2error[:, t] = np.dot(k, np.transpose(errorTraceInt)) * wErrFeed * -10

        # accumulate output for time t
        output[:, t] = np.dot(np.dot(k, np.transpose(spikesR)), wSynOut)
        
        # calculate error for time t
        for out in range(dimOutput):
            error[out, t] = output[out, t] - target[out, t]
            #error[out, t] = target[out, t] - output[out, t]
        
        # calculate soma current J by adding gain and bias
        iL1 = gainL1 * iL1 + biasL1
        iL2 = gainL2 * iL2 + biasL2

        # integration
        spikesL1 = np.roll(spikesL1, -1, axis=1)
        spikesL2 = np.roll(spikesL2, -1, axis=1)
        (vL1, refractoryL1, spikesL1[:,tau_s-1]) = lif_neuron(vL1, refractoryL1, iL1)
        (vL2, refractoryL2, spikesL2[:,tau_s-1]) = lif_neuron(vL2, refractoryL2, iL2)

        if t<=tLearn:
            # weight updates: wFFsynL2, wRsynL2, wSynOut
            errorTrace = np.append(np.zeros([dimOutput, max(0, tau_learn - t)]), error[:, max(0, t-tau_learn):t], axis=1)
            #wFFsynL2 = wFFsynL2 - eta * np.dot(np.transpose(np.dot(k, np.transpose(spikes))), np.transpose(it2error[:, t]))
            wRsynL2 = wRsynL2 - eta * np.dot(kLearn, np.transpose(errorTrace)) * np.dot(np.dot(spikesR, np.transpose(k)), wErrFeed)
            #np.dot(np.transpose(np.dot(k, np.transpose(spikesR))), np.transpose(it2error[:, t].reshape(nNeuronsL2, 1)))

            #print(np.shape(wSynOut))
            #wSynOut = wSynOut - eta * np.dot(kLearn, np.transpose(errorTrace)) * np.transpose(wErrFeed) * np.dot(spikesR, np.transpose(k))
            wSynOut = wSynOut - eta * np.dot(kLearn, np.transpose(errorTrace)) * np.dot(spikesR, np.transpose(k))
        
        # spike monitors
        spikeMonL1[0:nMonN,t] = spikesL1[monNL1,tau_s-1]
        spikeMonL2[0:nMonN,t] = spikesL2[monNL2,tau_s-1]

        # weight monitor
        wtMonRec[0:nMonW, t] = wRsynL2[monSrcN,monDstN]
        wtMonOut[0:nMonW, t] = wSynOut[monSrcN,0]

    ## end of simulation step loop

    # Simultion over - plot results
    plt.plot(target[0,:], label="Target") # blue
    plt.plot(output[0,:], label="Prediction") # orange
    #plt.show()
    # plt.plot(target[0,:])
    plt.plot(error[0,:], label="Error")
    #plt.plot(np.mean(it2ffr, axis=0)) # green
    #plt.plot(np.mean(it2error, axis=0)) # red
    plt.plot(np.zeros(tSim),'r-')
    plt.legend()
    plt.show()    

    fig, axarr = plt.subplots(2,1)
    fig.suptitle("Raster plot of neuron spikes")
    
    axarr[0].plot(range(0,tSim),np.transpose(spikeMonL1[:,:])*range(1,nMonN+1),'b.')
    axarr[0].set_title('L1 neurons')

    axarr[1].plot(range(0,tSim),np.transpose(spikeMonL2[:,:])*range(1,nMonN+1),'b.')
    axarr[1].set_title('L2 neurons')
    plt.show()

    fig, axarr = plt.subplots(2,1)
    fig.suptitle("Evolution of weights")

    axarr[0].plot(range(0,tSim),np.transpose(wtMonRec[:,:]),'.')
    axarr[0].set_title('Recurrent weights')
    axarr[1].plot(range(0,tSim),np.transpose(wtMonOut[:,:]),'.')
    axarr[1].set_title('Readout weights')
    plt.show()

    plt.plot(iMonFFR, label="feed-forward and recurrent current")
    plt.plot(iMonErr, label="Negatively modulated error current")
    plt.title("Average input current components to Layer 2")
    plt.plot(np.zeros(tSim),'r-')
    plt.legend()
    plt.show()

    print('total L2 spikes:', np.sum(spikesL2))
    print('mean ff weight: ', np.mean(wFFsynL2))
    print('mean recurrent weight: ', np.mean(wRsynL2))
    print('mean abs error: ', np.mean(np.abs(error[0,:])))