import cirq
import cirq_google

line_qubits = cirq.LineQubit.range(3)
grid_qubits = cirq.GridQubit.rect(1, 3)
dev = cirq_google.devices.XmonDevice(None, None, None, grid_qubits)
circuit = cirq.Circuit(
    cirq.X(grid_qubits[0])**0.5,
    cirq.CX(grid_qubits[0], grid_qubits[1]),
    cirq.CX(grid_qubits[1], grid_qubits[2]),
    cirq.decompose(cirq.CX(grid_qubits[0], grid_qubits[2])),
    cirq.measure(grid_qubits[2], key='m'),
    device=dev
)
print(circuit)
