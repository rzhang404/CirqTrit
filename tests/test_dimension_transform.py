import cirq
from transformations.dimension_transform import qubit_to_qutrit, qutrit_to_qubit
import pytest


def test_dimension_transform():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.Z(qubits[1]),
        cirq.Y(cirq.LineQubit(4)),
    )
    circtrit = qubit_to_qutrit(circuit)
    revcirc = qutrit_to_qubit(circtrit)
    assert circuit == revcirc
