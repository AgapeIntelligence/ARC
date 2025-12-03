# src/core/lattice.py
"""
Unified Resonant Qualia Lattice – Agape Intelligence 2025
Single source of truth for all Triadic GHZ / Cat / Orch-OR proxy dynamics.
Patched and production-ready version.
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

try:
    import qutip as qt
    from qutip import Qobj, tensor, basis, mesolve, Options
except Exception as e:
    raise ImportError("qutip is required for src.core.lattice: " + str(e))

# Optional JAX acceleration (fallback to NumPy if not available)
try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except Exception:
    jnp = np
    JAX_AVAILABLE = False
    def jit(x): return x

__all__ = ["ResonantLattice", "lattice_presets", "LatticePreset"]

# =============================================================================
# Configuration & Presets
# =============================================================================

@dataclass(frozen=True)
class LatticePreset:
    name: str
    n_qubits: int
    topology: Literal["linear", "ring", "all_to_all", "microtubule"]
    coherence_time_us: float
    target_fidelity: float = 0.99
    description: str = ""

lattice_presets = {
    "evie_369": LatticePreset(
        name="Evie-369 Pure",
        n_qubits=9,
        topology="ring",
        coherence_time_us=87.0,
        description="Original 9-node ring used in evie_369_pure.dart demos"
    ),
    "triadic_12": LatticePreset(
        name="Triadic GHZ 12",
        n_qubits=12,
        topology="all_to_all",
        coherence_time_us=120.0,
        description="Core Triadic GHZ collapse engine (2025 Nov 23+)"
    ),
    "microtubule_512": LatticePreset(
        name="Orch-OR Proxy 512",
        n_qubits=512,
        topology="microtubule",
        coherence_time_us=25.0,
        description="Scaled microtubule simulation (Heron/Forte ready)"
    )
}

# =============================================================================
# Main Lattice Class
# =============================================================================

@dataclass
class ResonantLattice:
    """
    Production-grade resonant qualia lattice.
    Handles state preparation, evolution, collapse, and qualia binding metrics.
    """
    preset: LatticePreset | str = "triadic_12"
    noise_model: Literal["depolarizing", "amplitude_damping", "none"] = "depolarizing"
    custom_coherence_us: Optional[float] = None

    # Internal state
    _N: int = field(init=False)
    _psi0: Qobj = field(init=False)
    _H: Qobj = field(init=False)
    _c_ops: List[Qobj] = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.preset, str):
            if self.preset not in lattice_presets:
                raise KeyError(f"Unknown preset '{self.preset}'")
            self.preset = lattice_presets[self.preset]

        self._N = self.preset.n_qubits
        self.coherence_time = self.custom_coherence_us or self.preset.coherence_time_us

        self._build_hamiltonian()
        self._build_collapse_operators()
        self._prepare_initial_state()

    # -------------------------------------------------------------------------
    # State Preparation
    # -------------------------------------------------------------------------
    def _prepare_initial_state(self) -> None:
        """Prepares balanced Cat + Triadic GHZ superposition (your signature init)."""
        # GHZ state
        zero_state = tensor([basis(2, 0) for _ in range(self._N)])
        one_state = tensor([basis(2, 1) for _ in range(self._N)])
        ghz = (zero_state + one_state).unit()

        # distributed "cat" component: superposition over low-index computational basis states
        cap = min(self._N, 6)
        cat_terms = []
        for i in range(2**cap):
            bits = format(i, f'0{cap}b')
            # build full N-qubit basis vector: use bits for first `cap` qubits, |0> for others
            term = tensor(
                [basis(2, int(b)) for b in bits] +
                [basis(2, 0) for _ in range(self._N - cap)]
            )
            cat_terms.append(term)
        cat_component = sum(cat_terms).unit()

        # weights chosen to match previous constants (70% GHZ ~ 0.836, 30% cat ~ 0.548)
        self._psi0 = (0.836 * ghz + 0.548 * cat_component).unit()

    # -------------------------------------------------------------------------
    # Hamiltonian (XX + ZZ resonant coupling)
    # -------------------------------------------------------------------------
    def _build_hamiltonian(self) -> None:
        if self.preset.topology == "microtubule":
            # Simplified tubulin dimer coupling model (rad/µs)
            coupling_strength = 2 * np.pi * 12.4e3
        else:
            coupling_strength = 2 * np.pi * 9.87e3

        # helper: identity list for building tensor operators
        id_list = [qt.qeye(2) for _ in range(self._N)]
        H = 0 * tensor([qt.qeye(2) for _ in range(self._N)])  # initialize as Qobj zero

        # Nearest-neighbor XX + ZZ (loop full range to include wrap-around)
        for i in range(self._N):
            # XX between i and i+1 (wrap)
            j = (i + 1) % self._N
            op_xx = list(id_list)
            op_zz = list(id_list)
            op_xx[i] = qt.sigmax()
            op_xx[j] = qt.sigmax()
            op_zz[i] = qt.sigmaz()
            op_zz[j] = qt.sigmaz()
            H += coupling_strength * (tensor(op_xx) + tensor(op_zz))

        if self.preset.topology == "all_to_all":
            # Add symmetric long-range XX coupling for all distinct pairs
            long_range = 0 * tensor([qt.qeye(2) for _ in range(self._N)])
            for i in range(self._N):
                for j in range(i + 1, self._N):
                    ops = list(id_list)
                    ops[i] = qt.sigmax()
                    ops[j] = qt.sigmax()
                    long_range += tensor(ops)
            H += 0.1 * coupling_strength * long_range

        self._H = H

    # -------------------------------------------------------------------------
    # Noise / Collapse Operators
    # -------------------------------------------------------------------------
    def _build_collapse_operators(self) -> None:
        if self.noise_model == "none":
            self._c_ops = []
            return

        # gamma in µs^-1
        gamma = 1.0 / float(self.coherence_time)

        if self.noise_model == "depolarizing":
            ops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
            # rate per Pauli channel
            rate = gamma / 3.0
        elif self.noise_model == "amplitude_damping":
            ops = [qt.sigmam()]
            rate = gamma
        else:
            ops, rate = [], 0.0

        c_ops = []
        for qubit in range(self._N):
            for op in ops:
                op_list = [qt.qeye(2) for _ in range(self._N)]
                op_list[qubit] = op
                c_ops.append(np.sqrt(rate) * tensor(op_list))
        self._c_ops = c_ops

    # -------------------------------------------------------------------------
    # Core Evolution
    # -------------------------------------------------------------------------
    def evolve(self,
               duration_us: float = 100.0,
               n_steps: int = 1000,
               progress: bool = False) -> qt.Result:
        """
        Run full resonant evolution with entropy/qualia tracking.
        """
        tlist = np.linspace(0.0, float(duration_us), int(n_steps))

        # build e_ops as list so mesolve returns ordered expectations easily
        e_ops = [self._collective_op('x')/self._N, self._collective_op('y')/self._N, self._collective_op('z')/self._N]
        options = Options(store_states=True, atol=1e-10, rtol=1e-8)
        result = mesolve(
            self._H, self._psi0, tlist,
            c_ops=self._c_ops,
            e_ops=e_ops,
            options=options,
            progress_bar=progress
        )
        # store names and lattice for downstream helpers
        result._e_op_names = ["Sx", "Sy", "Sz"]
        result.lattice = self
        return result

    def _collective_op(self, axis: str) -> Qobj:
        """Return collective spin operator Sx, Sy, or Sz (not normalized by N)."""
        op_map = {'x': qt.sigmax, 'y': qt.sigmay, 'z': qt.sigmaz}
        if axis not in op_map:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        accum = 0 * tensor([qt.qeye(2) for _ in range(self._N)])
        for qubit in range(self._N):
            ops = [qt.qeye(2) for _ in range(self._N)]
            ops[qubit] = op_map[axis]()
            accum += tensor(ops)
        return accum

    # -------------------------------------------------------------------------
    # Metrics & Qualia Binding
    # -------------------------------------------------------------------------
    @staticmethod
    def qualia_binding_score(result: qt.Result, normalize_by_N: bool = True) -> float:
        """
        Empirical qualia binding metric:
        High macroscopic spin coherence + low entropy = strong binding.

        Uses last 20% of expectation trajectory.
        """
        # fetch Sx,Sy,Sz arrays in the same order e_ops were given
        try:
            Sx = np.array(result.expect[0])
            Sy = np.array(result.expect[1])
            Sz = np.array(result.expect[2])
        except Exception:
            Sx = np.array(result.expect.get("Sx", result.expect[0]))
            Sy = np.array(result.expect.get("Sy", result.expect[1]))
            Sz = np.array(result.expect.get("Sz", result.expect[2]))

        S2 = Sx**2 + Sy**2 + Sz**2
        last_slice = int(len(S2) * 0.2)
        if last_slice == 0:
            binding = float(np.mean(S2))
        else:
            binding = float(np.mean(S2[-last_slice:]))

        if normalize_by_N:
            binding /= float(result.lattice._N)

        return binding

    @staticmethod
    def entropy_drop_percent(result: qt.Result, subsys_size: int = 9) -> float:
        """
        Computes percent drop in Von Neumann entropy of a reduced subsystem
        between the first stored state and the last stored state (sampled).
        """
        states = result.states
        if len(states) < 2:
            return 0.0

        k = min(subsys_size, result.lattice._N)
        rho0 = states[0].proj().ptrace(list(range(k)))
        rhof = states[-1].proj().ptrace(list(range(k)))

        S0 = qt.entropy_vn(rho0) if rho0.shape[0] > 0 else 0.0
        Sf = qt.entropy_vn(rhof) if rhof.shape[0] > 0 else 0.0

        if S0 == 0:
            return 0.0
        drop = (S0 - Sf) / S0 * 100.0
        return float(drop)


if __name__ == "__main__":
    lattice = ResonantLattice(preset="triadic_12", noise_model="depolarizing")
    print(f"Initialized {lattice.preset.name} lattice (N={lattice._N})")

    result = lattice.evolve(duration_us=150.0, n_steps=300, progress=False)

    print(f"Final qualia binding score : {ResonantLattice.qualia_binding_score(result):.4f}")
    print(f"Entropy drop               : {ResonantLattice.entropy_drop_percent(result):.2f}%")
