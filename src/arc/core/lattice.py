"""
ARC Unified Lattice Core — with HEOM extension hook.
"""

from arc.heom.heom_engine import HEOMEngine

class LatticeCore:
    def __init__(self, dim=4, heom_config=None):
        self.dim = dim
        self.heom = HEOMEngine(heom_config or {})

    def evolve(self, rho):
        return self.heom.step(rho)
