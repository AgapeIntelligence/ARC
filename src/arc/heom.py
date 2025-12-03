from typing import List
import jax
import jax.numpy as jnp
from jax import jit
import qutip as qt

class HEOMSolver:
def init(self, H, Q, lam, gamma, beta, K=48, depth=8):
self.H = H
self.Q = Q
self.lam = lam
self.gamma = gamma
self.beta = beta
self.K = K
self.depth = depth
self.dim = H.shape[0]
self._build_matsubara()

def _build_matsubara(self):
    k = jnp.arange(1, self.K+1)
    nu_k = 2 * jnp.pi * k / self.beta
    self.ck = self.lam * self.gamma / (nu_k + 1j*self.gamma)
    self.vk = nu_k

@jit
def _liouvillian(self, rho_vec):
    L = jnp.zeros((self.dim**2, self.dim**2), dtype=jnp.complex128)
    L = L.at[:].add(jax.scipy.sparse.linalg.kron(jnp.eye(self.dim**2), self.H.full()) -
                    jax.scipy.sparse.linalg.kron(self.H.full().T, jnp.eye(self.dim**2)))
    zero_term = self.lam * self.gamma
    for ck in jnp.concatenate([jnp.array([zero_term]), self.ck]):
        L = L - 1j*ck*(jax.scipy.sparse.linalg.kron(self.Q.full().T.conj(), self.Q.full()) -
                        0.5*(jax.scipy.sparse.linalg.kron(jnp.eye(self.dim), (self.Q.dag()*self.Q).full()) +
                             jax.scipy.sparse.linalg.kron((self.Q.dag()*self.Q).full().T, jnp.eye(self.dim))))
    return L @ rho_vec

def solve(self, rho0, tlist):
    rho_vec0 = rho0.full().ravel()
    solver = jax.scipy.integrate.solve_ivp(
        lambda t, y: self._liouvillian(y),
        [tlist[0], tlist[-1]],
        rho_vec0,
        t_eval=tlist,
        method='BDF',
        atol=1e-10,
        rtol=1e-8
    )
    return [qt.Qobj(rho.reshape(self.dim, self.dim)) for rho in solver.y.T]


