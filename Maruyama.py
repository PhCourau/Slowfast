import numpy as np

def Simulate_EM(om2,eta,alpha,theta,N2,L,dt,T,X0=None,record_every=100):
  """Simulates a multilocus WF diffusion from Euler-Maruyama scheme
  om2: inverse selection strength
  alpha: (np array shape L) effect size
  theta: (np array shape (2,L)) mutation rates to (resp. from) the trait-increasing allele. A vector
  N2: 2 * population size (diploid)
  L: number of loci
  dt: precision
  T: duration (in units of N2)
  X0: starting population (if None, uniform random variables are taken)
  record_every : how often should we record the value of X ?
  """
  if X0 is None:
    X0 = np.random.random(L)

  number_iterations = int(T/dt)
  listX = np.zeros((number_iterations//record_every+1,L))
  listX[0] = X0
  oldX = X0
  for t in range(1,number_iterations):
    dmutation = np.sum(theta * np.array([1-oldX,-oldX]),axis=0) *dt
    dselection = (alpha/om2 * (eta-2*alpha@oldX) +  alpha**2/om2 *(oldX-1/2))*dt #Appendix A.1
    ddrift = np.random.normal(size=L)*np.sqrt(oldX*(1-oldX)*dt/N2)
    newX = oldX + dmutation + dselection + ddrift
    newX = np.max([np.zeros(L),np.min([newX,np.ones(L)],axis=0)],axis=0)
    if t%record_every == 0:
      listX[t//record_every] = newX
    if (t*10)%number_iterations==0:
      print("Done: "+ str((t*100)//number_iterations) + " per cent")
    oldX = newX
  return listX
