import cirq
from ops.to_qutrit_wrappers import SingleQubitGateToQutritGate, TwoQubitGateToQutritGate


@cirq.transformer
def qubit_to_qutrit(circuit: cirq.AbstractCircuit, *, context=None) -> cirq.Circuit:
    """Takes a circuit operating over qubits,
    exchanges qubits for equivalent qutrits,
    and wraps all qubit gates in qutrit gates.

    Args:
        circuit: The circuit to transform dimensions for.
        context: Transformer context. (unused, included for decorator API)
    """

    def wrap_gate(op):
        qubits = op.qubits
        q_type = type(qubits[0])
        if len(qubits) == 1:
            gate = SingleQubitGateToQutritGate(op.gate)
            if q_type is cirq.LineQubit:
                return gate.on(cirq.LineQid(qubits[0].x, dimension=3))
            elif q_type is cirq.NamedQubit:
                return gate.on(cirq.NamedQid(qubits[0].name, dimension=3))
            elif q_type is cirq.GridQubit:
                return gate.on(cirq.GridQid(qubits[0].row, qubits[0].col, dimension=3))

        elif len(qubits) == 2:
            gate = TwoQubitGateToQutritGate(op.gate)
            if q_type is cirq.LineQubit:
                return gate.on(
                    cirq.LineQid(qubits[0].x, dimension=3), cirq.LineQid(qubits[1].x, dimension=3)
                )
            elif q_type is cirq.NamedQubit:
                return gate.on(
                    cirq.NamedQid(qubits[0].name, dimension=3),
                    cirq.NamedQid(qubits[1].name, dimension=3),
                )
            elif q_type is cirq.GridQubit:
                return gate.on(
                    cirq.GridQid(qubits[0].row, qubits[0].col, dimension=3),
                    cirq.GridQid(qubits[1].row, qubits[1].col, dimension=3),
                )

        # If reached, either gate acts on more than 2 qubits or
        # the type of qubit used is not supported
        raise TypeError

    batch_replace = []
    for i, op in circuit.findall_operations(lambda x: True):
        batch_replace.append((i, op, wrap_gate(op)))
    transformed_circuit = circuit.unfreeze(copy=True)
    transformed_circuit.batch_replace(batch_replace)
    return transformed_circuit


@cirq.transformer
def qutrit_to_qubit(circuit: cirq.AbstractCircuit, *, context=None) -> cirq.Circuit:
    """Takes a circuit operating over qutrits,
    exchanges qutrits for equivalent qubits,
    and wraps all qutrit gates in qubit gates.

    Args:
        circuit: The circuit to transform dimensions for.
        context: Transformer context. (unused, included for decorator API)
    """

    def unwrap_gate(op):
        qutrits = op.qubits
        wrapped_gate = op.gate
        q_type = type(qutrits[0])
        if type(wrapped_gate) == SingleQubitGateToQutritGate:
            base_gate = wrapped_gate.base_gate
            if q_type is cirq.LineQid:
                return base_gate.on(cirq.LineQubit(qutrits[0].x))
            elif q_type is cirq.NamedQid:
                return base_gate.on(cirq.NamedQubit(qutrits[0].name))
            elif q_type is cirq.GridQid:
                return base_gate.on(cirq.GridQubit(qutrits[0].row, qutrits[0].col))
        elif type(wrapped_gate) == TwoQubitGateToQutritGate:
            base_gate = wrapped_gate.base_gate
            if q_type is cirq.LineQid:
                return base_gate.on(cirq.LineQubit(qutrits[0].x), cirq.LineQubit(qutrits[1].x))
            elif q_type is cirq.NamedQid:
                return base_gate.on(
                    cirq.NamedQubit(qutrits[0].name), cirq.NamedQubit(qutrits[1].name)
                )
            elif q_type is cirq.GridQid:
                return base_gate.on(
                    cirq.GridQubit(qutrits[0].row, qutrits[0].col),
                    cirq.GridQubit(qutrits[1].row, qutrits[1].col),
                )

        # If reached, either gate acts on more than 2 qutrits,
        # the gate isn't actually a wrapped gate, or
        # the type of qid used is not supported
        raise TypeError

    batch_replace = []
    for i, op in circuit.findall_operations(lambda x: True):
        batch_replace.append((i, op, unwrap_gate(op)))
    transformed_circuit = circuit.unfreeze(copy=True)
    transformed_circuit.batch_replace(batch_replace)
    return transformed_circuit
