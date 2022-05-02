import cirq

# Make unconstrained input circuit
num_input_qubits = 10
circuit_depth = 1
op_density = 0.5  # probability at each moment a qubit is to have a gate acting on it
gate_domain = {
    cirq.X: 1,
    cirq.Y: 1,
    cirq.Z: 1,
    cirq.H: 1,
    cirq.S: 1,
    cirq.T: 1,
    cirq.CNOT: 2,
    cirq.CZ: 2,
    cirq.SWAP: 2,
    cirq.ISWAP: 2,
    # cirq.CZPowGate: 2,  # Somehow causing problems rn
    # cirq.TOFFOLI: 3,  # Not supported by tket
}
in_circ = cirq.testing.random_circuit(
    qubits=num_input_qubits, n_moments=circuit_depth, op_density=op_density, gate_domain=gate_domain
)

reps = 3
non_noisy_simulator = cirq.DensityMatrixSimulator()
# in_circ.append(cirq.H(q) for q in in_circ.all_qubits())
print(
    non_noisy_simulator.simulate_expectation_values(
        in_circ, observables=[cirq.Z(q) for q in in_circ.all_qubits()]
    )
)
print(cirq.final_state_vector(in_circ))

# in_circ.append([cirq.ops.measure(*in_circ.all_qubits(), key='z')])


result = non_noisy_simulator.simulate(in_circ)
print(result.final_density_matrix)

print(
    cirq.fidelity(
        result.final_density_matrix,
        result.final_density_matrix,
        qid_shape=(len(result.final_density_matrix),),
    )
)


noisy_simulator = cirq.DensityMatrixSimulator(noise=cirq.DepolarizingChannel(0.1))
noisy_result = noisy_simulator.simulate(in_circ)
print(noisy_result.final_density_matrix)

print(
    cirq.fidelity(
        result.final_density_matrix,
        noisy_result.final_density_matrix,
        qid_shape=(len(result.final_density_matrix),),
    )
)
