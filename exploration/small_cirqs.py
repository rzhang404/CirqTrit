import cirq

line_qubits = cirq.LineQubit.range(3)

abstract_circuit = cirq.Circuit(
    # cirq.X(line_qubits[0])**0.5,
    # cirq.CX(line_qubits[0], line_qubits[1]),
    # cirq.CX(line_qubits[1], line_qubits[2]),
    # cirq.CX(line_qubits[0], line_qubits[2]), # long-range control?
    # cirq.measure(line_qubits[2], key='m')
    cirq.TOFFOLI(line_qubits[0], line_qubits[1], line_qubits[2])
)

print("Circuit A:")
print(abstract_circuit)

# Decomposition to look at overriding
decomp_circ = cirq.Circuit(cirq.decompose(cirq.TOFFOLI(line_qubits[0], line_qubits[1], line_qubits[2])))

print("Circuit B:")
print(decomp_circ)

# Simulation syntax
simulator = cirq.DensityMatrixSimulator()
abs_result = simulator.simulate(abstract_circuit)
decomp_result = simulator.simulate(decomp_circ)


print(cirq.fidelity(abs_result.final_density_matrix, decomp_result.final_density_matrix, qid_shape=(len(abs_result.final_density_matrix),)))
print(cirq.fidelity(decomp_result.final_density_matrix, decomp_result.final_density_matrix, qid_shape=(len(abs_result.final_density_matrix),)))
