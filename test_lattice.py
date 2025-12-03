# tests/test_lattice.py
import numpy as np
import pytest

def test_mean_field_shapes():
    from src.core.mean_field import MeanFieldLattice
    mf = MeanFieldLattice(n_qubits=10, topology="ring", dt=0.05)
    traj = mf.run(total_steps=5)
    assert traj.shape == (5, 10, 3)
    # vectors should remain normalized (within tolerance)
    norms = np.linalg.norm(traj[-1], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_qutip_small_smoke():
    # This test exercises the full ResonantLattice for a tiny N to ensure it runs.
    # It will be skipped if qutip isn't available.
    pytest.importorskip("qutip")
    from src.core.lattice import ResonantLattice
    lattice = ResonantLattice(preset="evie_369", noise_model="none")  # 9 qubits preset
    # run a very short evolution to smoke test (small n_steps to keep time low)
    result = lattice.evolve(duration_us=1.0, n_steps=10, progress=False)
    # Expect result to have states and expect arrays
    assert hasattr(result, "states")
    assert len(result.states) == 10
    assert len(result.expect) == 3  # Sx, Sy, Sz
