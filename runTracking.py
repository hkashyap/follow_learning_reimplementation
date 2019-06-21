# This modeule runs the network in learning and 
# testing phases, handles input and output

#%%
import matplotlib.pyplot as plt
from netConfig import *

# simulation parameters
tLearn = 20
# tTest = 100
# target characterstics
freq = 1
peak = 6

# generate random spikes - 1 for spike
#spike_train_ff = np.random.random_integers(0, 1, size=(nSynFF,tLearn))

# generate command signal of dimension dimCommand
# command = np.append(np.ones(int(tLearn/2)) * np.random.random(), np.ones(int(tLearn/2)) * np.random.random())
command = np.append(np.ones(int(tLearn/2)) * 0.9, np.ones(int(tLearn/2)) * 0.9)

# to record output
vt1 = np.zeros(tLearn)
it1 = np.zeros(tLearn)
vt2 = np.zeros(tLearn)
it2 = np.zeros(tLearn)
it2ffr = np.zeros([nNeuronsL2, tLearn])
it2error = np.zeros([nNeuronsL2, tLearn])

# spike tables
spikesL1 = np.zeros([nNeuronsL1, tLearn])
spikesL2 = np.zeros([nNeuronsL2, tLearn])

# output
output = np.zeros([dimOutput, tLearn])
error = np.zeros([dimOutput, tLearn])

# target
# tLearn = 1000
# freq = 1
target = np.zeros([dimOutput, tLearn])
for dim in range(dimOutput):
    for t in range(tLearn):
        target[dim, t] = np.sin(np.pi/2 + (t*freq/1000)*2*np.pi)*peak
# plt.plot(target[0,:])
# plt.show()

if __name__=='__main__':
    # run the learning phase
    for t in range(tLearn):
        # L1 neurons
        for nPost in range(nNeuronsL1):
            #find total input current to the neuron
            i = 0
            for comm in range(dimCommand):
                i =  i + command[t] * wSynCmdL1[comm, nPost]
            i = i * gainL1[nPost] + biasL1[nPost] # multiply by cell specific membrane resistance Rm
            it1[t] = i
            (vL1[nPost], refractoryL1[nPost], spikesL1[nPost,t]) = lif_neuron(vL1[nPost], refractoryL1[nPost], i)
            vt1[t] = vL1[nPost]

        # L2 neurons
        for nPost in range(nNeuronsL2):
            #find total input current to the neuron
            i = 0            
            # feed-forward connections
            for nPre in range(nNeuronsL1):
                spikes = np.append(np.zeros(max(0, tau_s - t)), spikesL1[nPre, max(0, t-tau_s):t])
                i =  i + np.convolve(spikes, k, 'valid') * wFFsynL2[nPre, nPost]
            
            # recurrent connections
            for nPre in range(nNeuronsL2):
                spikesR = np.append(np.zeros(max(0, tau_s - t)), spikesL2[nPre, max(0, t-tau_s):t])
                i =  i + np.convolve(spikesR, k, 'valid') * wRsynL2[nPre, nPost]
            
            it2ffr[nPost, t] = i

            # error feedback
            for out in range(dimOutput):
                avgErr = np.append(np.zeros(max(0, tau_s - t)), error[out, max(0, t-tau_s):t])
                i = i + np.convolve(avgErr, k, 'valid') * wErrFeed[out, nPost] * -5
                #i = i + error[out,max(0, t-1)] * wErrFeed[out, nPost] * -5
            
            #it2error[t] = error[out,max(0, t-1)] * wErrFeed[out, nPost] * -5
            it2error[nPost, t] = np.convolve(avgErr, k, 'valid') * wErrFeed[out, nPost] * -5

            #i = i * rmL2[nPost] * (np.random.randn(1)*0.2 + 0.5)
            i = i *gainL2[nPost] + biasL2[nPost]
            it2[t] = i
            (vL2[nPost], refractoryL2[nPost], spikesL2[nPost,t]) = lif_neuron(vL2[nPost], refractoryL2[nPost], i)
            vt2[t] = vL2[nPost]

            # accumulate output for time t
            for out in range(dimOutput):
                spikesOut = np.append(np.zeros(max(0, tau_s - t)), spikesL2[nPost, max(0, t-tau_s):t])
                output[out, t] = output[out, t] + np.convolve(spikesOut, k, 'valid') * wSynOut[nPost, out]
        
        # calculate error for time t
        for out in range(dimOutput):
            error[out, t] = output[out, t] - target[out, t]

    # plt.plot(it1)
    # plt.plot(vt1)
    # plt.plot(np.mean(spikesL1, axis=0))
    # plt.show()
    # print('Total L1 spikes:', np.sum(spikesL1))
    # print('mean input weight: ', np.mean(wSynCmdL1))

    #plt.plot(it2)
    plt.plot(np.mean(it2error, axis=0))
    #plt.show()
    #plt.plot(it2/rmL2[99])
    plt.plot(np.mean(it2ffr, axis=0))
    #plt.show()
    #plt.plot(spikesL2[99,:])
    #plt.plot(vt2)
    plt.plot(np.sum(spikesL2, axis=0))
    #plt.show()
    plt.plot(error[0,:])
    plt.show()    
    print('total L2 spikes:', np.sum(spikesL2))
    print('mean ff weight: ', np.mean(wFFsynL2))
    print('mean recurrent weight: ', np.mean(wRsynL2))

    #print(np.mean(it2ffr))
    # plt.plot(target[0,:])
    # plt.plot(output[0,:])
    # plt.plot(error[0,:])
    # plt.show()

# plt.plot(np.sum(spikesL2, axis=0))
# plt.show()

# t = 250
# print('e:', error[0, t])
#print(wErrFeed[0, 99], it2[t])

# plt.plot((it2 - it2ffr)/error[0,:])
# #plt.plot()
# plt.show()
    #plt.plot(error[0,:])
    #plt.plot(it2)
    # plt.plot(it2ffr)
    # plt.plot(it2error)
    # plt.plot(it2ffr + it2error)
    # #plt.plot(it2error/np.append(np.zeros(1), error[0,0:299]))
    # #print((it2error/np.append(np.ones(1), error[0,0:299])))
    # #plt.plot(vt2)
    # #plt.plot(np.mean(spikesL2, axis=0))
    #plt.show()    
    # print(np.mean(it2error/np.append(np.zeros(1), error[0,0:299])))
    