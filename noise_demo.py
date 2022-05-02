import cirq
import numpy as np
from noise_models.hardware_aware import HardwareAwareSymmetricNoise
from ops.to_qutrit_wrappers import SingleQubitGateToQutritGate, TwoQubitGateToQutritGate
from ops.ternary_gates import (
    QutritPlusGate,
    OneControlledPlusGate,
    TwoControlledPlusGate,
    QutritMinusGate,
    OneControlledMinusGate,
    TwoControlledMinusGate,
)


# Make unconstrained input circuit
qutrits = [cirq.NamedQid(str(i), dimension=3) for i in range(8)]
circuit_depth = 6
op_density = 0.3  # probability at each moment a qubit is to have a gate acting on it
noise_simulable_gate_domain = {
    SingleQubitGateToQutritGate(cirq.X): 1,
    SingleQubitGateToQutritGate(cirq.Y): 1,
    SingleQubitGateToQutritGate(cirq.Z): 1,
    SingleQubitGateToQutritGate(cirq.H): 1,
    SingleQubitGateToQutritGate(cirq.S): 1,
    SingleQubitGateToQutritGate(cirq.T): 1,
    TwoQubitGateToQutritGate(cirq.CNOT): 2,
    TwoQubitGateToQutritGate(cirq.CZ): 2,
    TwoQubitGateToQutritGate(cirq.SWAP): 2,
    TwoQubitGateToQutritGate(cirq.ISWAP): 2,
    # QutritPlusGate: 1,
    # OneControlledPlusGate: 2,
    # TwoControlledPlusGate: 2,
    # QutritMinusGate: 1,
    # OneControlledMinusGate: 2,
    # TwoControlledMinusGate: 2,
}
circuit = cirq.testing.random_circuit(
    qubits=qutrits,
    n_moments=circuit_depth,
    op_density=op_density,
    gate_domain=noise_simulable_gate_domain,
)

print("Generated circuit:", circuit)

# Build error weights
edges = [(qutrits[0], qutrits[1]), (qutrits[1], qutrits[2])]
single_qutrit_error_dict = dict()
two_qutrit_error_dict = dict()
p_1 = 0.001 / 3
p_2 = 0.01 / 15
for q in qutrits:
    single_qutrit_error_dict[q] = np.random.uniform(0, p_1)  # initialize single body gate errors
    two_qutrit_error_dict[q] = dict()  # initialize next level of dicts
for e in edges:
    edge_error = np.random.uniform(0, p_2)
    two_qutrit_error_dict[e[0]][e[1]] = edge_error
    two_qutrit_error_dict[e[1]][e[0]] = edge_error  # set two-body gate errors for both directions

noise_model = HardwareAwareSymmetricNoise(
    single_qutrit_hardware_error_rates=single_qutrit_error_dict,
    two_qutrit_hardware_error_rates=two_qutrit_error_dict,
    lambda_short=100.0 / 10000.0,
    lambda_long=300.0 / 10000.0,
)

ideal_simulator = cirq.DensityMatrixSimulator()
noisy_simulator = cirq.DensityMatrixSimulator(noise=noise_model)

ideal_result = ideal_simulator.simulate(circuit)
noisy_result = noisy_simulator.simulate(circuit)

print(
    "Resulting fidelity: ",
    cirq.qis.fidelity(
        ideal_result.final_density_matrix,
        noisy_result.final_density_matrix,
        (len(ideal_result.final_density_matrix),),
    ),
)
