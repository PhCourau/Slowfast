import numpy as np
import matplotlib.pyplot as plt

### Initial dynamics
L=1000
listX,alpha,theta,T = np.load("sims/L"+str(L)+".npy",allow_pickle=True)
gamma,N,alphabar,nbL,listL,eta,dt,T = np.load("sims/parameters.npy",allow_pickle=True)
vartheta = np.mean(theta)*2
listX=listX[:-1]
listt = np.linspace(0,T,np.shape(listX)[0])
tmax = 10000

#cutoffP = 0.01
#cutoffQ = 0.3


ax=plt.axes()
#ax.plot(listt,np.mean(listX<= cutoffP, axis=1),label=r"$P[X_t\leq "+str(cutoffP)+"]$")
#Q = np.mean(listX*(1-listX)*(listX<=cutoffQ),axis=1)/(N*vartheta)
#ax.plot(listt,Q,label=r"$Q[X_t\leq "+str(cutoffQ)+"]$")
ax.plot(listt[:tmax],(2*listX[-tmax:]@alpha-eta)*gamma,"o",label=r"$\tilde{\Delta}_t/\bar\alpha$",alpha=.05)
typical_locus = np.where(listX[-500]*(1-listX[-500])>=.1)[0][1]
ax.plot(listt[:tmax],listX[-tmax:,typical_locus],label=r"$X_t^0$")
ax.set_ylim([-1,1])
ax.set_xlabel(r"$t/N_e$")

ax.legend()
plt.show()

