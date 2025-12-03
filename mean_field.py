# src/core/mean_field.py
"""
Mean-field evolution for large-N resonant lattice.
Provides an O(N) classical approximation evolving Bloch vectors (single-qubit density matrices)
coupled via mean-field XX + ZZ interactions. Useful as a fast proxy when full Hilbert
space simulation is infeasible.

Equations (for qubit i Bloch vector r_i):
dr_i/dt = r_i × (B_eff_i) - gamma * (r_i - r_eq)
Where B_eff_i is the effective field from neighbors: proportional to sum_j (J_x * r_j.x, J_y * r_j.y, J_z * r_j.z)
This implementation uses a simple RK4 integrator and supports ring / all_to_all coupling.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional, List

@dataclass
class MeanFieldLattice:
    n_qubits: int = 64
    topology: Literal["ring", "all_to_all"] = "ring"
    Jx: float = 1.0
    Jz: float = 1.0
    damping: float = 0.0  # phenomenological relaxation
    dt: float = 0.01
    state: np.ndarray = field(init=False)  # shape (n_qubits, 3) Bloch vectors

    def __post_init__(self):
        # initialize near +x GHZ-like aligned Bloch vectors with small noise
        self.state = np.tile(np.array([1.0, 0.0, 0.0]), (self.n_qubits, 1))
        self.state += 0.01 * np.random.randn(self.n_qubits, 3)
        self.state = self._renormalize(self.state)

    def _renormalize(self, r: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(r, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return r / norms

    def _effective_field(self, idx: int, r: np.ndarray) -> np.ndarray:
        if self.topology == "all_to_all":
            others = np.mean(np.delete(r, idx, axis=0), axis=0)
        else:  # ring: neighbors only
            left = r[(idx - 1) % self.n_qubits]
            right = r[(idx + 1) % self.n_qubits]
            others = 0.5 * (left + right)
        # B_eff proportional to (Jx * sum_x, 0, Jz * sum_z)
        return np.array([self.Jx * others[0], 0.0, self.Jz * others[2]])

    def _drdt(self, r: np.ndarray) -> np.ndarray:
        dr = np.zeros_like(r)
        for i in range(self.n_qubits):
            B = self._effective_field(i, r)
            # r x B (cross product)
            dr[i] = np.cross(r[i], B) - self.damping * (r[i])
        return dr

    def step(self, steps: int = 1):
        """Advance the mean-field state by `steps` RK4 steps of size dt."""
        for _ in range(steps):
            r = self.state
            k1 = self._drdt(r)
            k2 = self._drdt(r + 0.5 * self.dt * k1)
            k3 = self._drdt(r + 0.5 * self.dt * k2)
            k4 = self._drdt(r + self.dt * k3)
            self.state = r + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            self.state = self._renormalize(self.state)

    def run(self, total_steps: int):
        traj = np.zeros((total_steps, self.n_qubits, 3))
        for t in range(total_steps):
            self.step(1)
            traj[t] = self.state.copy()
        return traj
