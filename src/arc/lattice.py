import qutip as qt
import numpy as np

def ghz_state(N: int):
return (qt.tensor([qt.basis(2,0)]*N) + qt.tensor([qt.basis(2,1)]*N)).unit()

def cat_state(N: int, alpha: float = 2.0):
c = qt.coherent(N, alpha) + qt.coherent(N, -alpha)
return c.unit()

def microtubule_hamiltonian(N: int, J: float = 2np.pi12.4e3):
H = 0
for n in range(N-1):
xx = qt.tensor([qt.sigmax() if i in (n,n+1) else qt.qeye(2) for i in range(N)])
zz = qt.tensor([qt.sigmaz() if i in (n,n+1) else qt.qeye(2) for i in range(N)])
H += J * (xx + zz)
return H
