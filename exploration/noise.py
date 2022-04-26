import cirq
import numpy as np
from prototype.noise_model import GokhaleNoiseModelOnQubits

qubits = cirq.LineQid.range(3, dimension=2)
circuit = cirq.Circuit(
    cirq.X(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1])
)

noise_model = cirq.AmplitudeDampingChannel(0.01)
noise_model = GokhaleNoiseModelOnQubits(0.09, 0.1)

sim = cirq.DensityMatrixSimulator(noise=noise_model)

dmat = sim.simulate(circuit).final_density_matrix

print(dmat, np.trace(dmat), np.isclose(np.trace(dmat), 1.0))