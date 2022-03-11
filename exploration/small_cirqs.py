import cirq

line_qubits = cirq.LineQubit.range(3)

abstract_circuit = cirq.Circuit(
    cirq.X(line_qubits[0])**0.5,
    cirq.CX(line_qubits[0], line_qubits[1]),
    cirq.CX(line_qubits[1], line_qubits[2]),
    cirq.CX(line_qubits[0], line_qubits[2]), # long-range control?
    cirq.measure(line_qubits[2], key='m')
)

print("Circuit A:")
print(abstract_circuit)

# Decomposition to look at overriding
decomp_circ = cirq.Circuit(cirq.decompose(cirq.TOFFOLI(line_qubits[0], line_qubits[1], line_qubits[2])))

print("Circuit B:")
print(decomp_circ)

# Simulation syntax
simulator = cirq.Simulator()
result = simulator.run(abstract_circuit, repetitions=20)
print("Results:")
print(result)
