import cirq
from ops.to_qutrit_wrappers import SingleQubitGateToQutritGate, TwoQubitGateToQutritGate
from ops.to_qubit_wrappers import SingleQutritGateToQubitGate, TwoQutritGateToQubitGate


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

        # Variables to store the new gate and its operands
        new_gate = None
        qutrits = None

        # Check if already was wrapped, returning to qutrit
        if (
            type(op.gate) == SingleQutritGateToQubitGate
            or type(op.gate) == TwoQutritGateToQubitGate
        ):
            new_gate = op.gate

        # Generate qutrits of the same type
        if len(qubits) == 1:
            new_gate = SingleQubitGateToQutritGate(op.gate)
            if q_type is cirq.LineQubit:
                qutrits = [cirq.LineQid(qubits[0].x, dimension=3)]
            elif q_type is cirq.NamedQubit:
                qutrits = [cirq.NamedQid(qubits[0].name, dimension=3)]
            elif q_type is cirq.GridQubit:
                qutrits = [cirq.GridQid(qubits[0].row, qubits[0].col, dimension=3)]

        elif len(qubits) == 2:
            new_gate = TwoQubitGateToQutritGate(op.gate)
            if q_type is cirq.LineQubit:
                qutrits = [
                    cirq.LineQid(qubits[0].x, dimension=3),
                    cirq.LineQid(qubits[1].x, dimension=3),
                ]
            elif q_type is cirq.NamedQubit:
                qutrits = [
                    cirq.NamedQid(qubits[0].name, dimension=3),
                    cirq.NamedQid(qubits[1].name, dimension=3),
                ]
            elif q_type is cirq.GridQubit:
                qutrits = [
                    cirq.GridQid(qubits[0].row, qubits[0].col, dimension=3),
                    cirq.GridQid(qubits[1].row, qubits[1].col, dimension=3),
                ]

        if new_gate is not None and qutrits is not None:
            return new_gate.on(*qutrits)

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

        # Variables to store the new gate and its operands
        new_gate = None
        qubits = None

        # Check if already was wrapped, returning to qutrit
        if (
            type(wrapped_gate) == SingleQubitGateToQutritGate
            or type(wrapped_gate) == TwoQubitGateToQutritGate
        ):
            new_gate = wrapped_gate.base_gate

        # Generate qubits of the same type
        if len(qutrits) == 1:
            if q_type is cirq.LineQid:
                qubits = [cirq.LineQubit(qutrits[0].x)]
            elif q_type is cirq.NamedQid:
                qubits = [cirq.NamedQubit(qutrits[0].name)]
            elif q_type is cirq.GridQid:
                qubits = [cirq.GridQubit(qutrits[0].row, qutrits[0].col)]
        elif len(qutrits) == 2:
            if q_type is cirq.LineQid:
                qubits = [cirq.LineQubit(qutrits[0].x), cirq.LineQubit(qutrits[1].x)]
            elif q_type is cirq.NamedQid:
                qubits = [cirq.NamedQubit(qutrits[0].name), cirq.NamedQubit(qutrits[1].name)]
            elif q_type is cirq.GridQid:
                qubits = [
                    cirq.GridQubit(qutrits[0].row, qutrits[0].col),
                    cirq.GridQubit(qutrits[1].row, qutrits[1].col),
                ]

        else:
            # Need to wrap this in a qubit gate
            if len(qutrits) == 1:
                new_gate = SingleQutritGateToQubitGate(op.gate)
            elif len(qutrits) == 2:
                new_gate = TwoQutritGateToQubitGate(op.gate)

        if new_gate is not None and qubits is not None:
            return new_gate.on(*qubits)

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
