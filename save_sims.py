from Maruyama import Simulate_EM
import numpy as np

gamma = 1
N=1
alphabar=1

dt=1e-6
T=5

nbL = 1
listL = np.arange(1,1+nbL)
listL = np.array([1000])
eta = 1.2*listL*alphabar
for (k,L) in enumerate(listL):
  alpha = np.random.exponential(alphabar,size=L)
  om2 = N *alphabar**2 /gamma
  theta = np.array([np.random.uniform(size=L)*.2,np.random.uniform(size=L)*.4])/(2*N)

  nbones = int(eta[k]//(2*alphabar))
  X0 = np.append(np.ones(nbones),np.zeros(L-nbones))
  listX = Simulate_EM(om2,eta[k],alpha,theta,2*N,L,dt,T,X0=X0)
  np.save("sims/L"+str(L)+".npy",np.array([listX,alpha,theta,T],dtype="object"))
  print("Done L="+str(L))

np.save("sims/parameters.npy",np.array([gamma,N,alphabar,nbL,listL,eta,dt,T],dtype="object"))
