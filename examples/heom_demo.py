from arc.core.lattice import LatticeCore

rho0 = [[1,0],[0,0]]  # Placeholder density matrix
core = LatticeCore(dim=2, heom_config={"gamma":0.1})
rho_final = core.evolve(rho0)

print("Initial:", rho0)
print("Final:  ", rho_final)
