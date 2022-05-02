import cirq
from ops.to_qutrit_wrappers import SingleQubitGateToQutritGate, TwoQubitGateToQutritGate


@cirq.transformer
def qubit_to_qutrit(
        circuit: cirq.AbstractCircuit, *, context=None
) -> cirq.Circuit:
    def wrap_gate(op):
        qubits = op.qubits
        q_type = type(qubits[0])
        if len(qubits) == 1:
            if q_type is cirq.LineQubit:
                gate = SingleQubitGateToQutritGate(op.gate)
                return gate.on(cirq.LineQid(qubits[0].x, dimension=3))
            elif q_type is cirq.NamedQubit:
                gate = SingleQubitGateToQutritGate(op.gate)
                return gate.on(cirq.NamedQid(qubits[0].name, dimension=3))
        elif len(qubits) == 2:
            gate = TwoQubitGateToQutritGate(op.gate)
            if q_type is cirq.LineQubit:
                return gate.on(cirq.LineQid(qubits[0].x, dimension=3),
                               cirq.LineQid(qubits[1].x, dimension=3))
            elif q_type is cirq.NamedQubit:
                return gate.on(cirq.NamedQid(qubits[0].name, dimension=3),
                               cirq.NamedQid(qubits[1].name, dimension=3))

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
def qutrit_to_qubit(
        circuit: cirq.AbstractCircuit, *, context=None
) -> cirq.Circuit:
    def unwrap_gate(op):
        qutrits = op.qubits
        base_gate = op.gate.base_gate
        q_type = type(qutrits[0])
        if len(qutrits) == 1:
            if q_type is cirq.LineQid:
                return base_gate.on(cirq.LineQubit(qutrits[0].x))
            elif q_type is cirq.NamedQid:
                return base_gate.on(cirq.NamedQubit(qutrits[0].name))
        elif len(qutrits) == 2:
            if q_type is cirq.LineQid:
                return base_gate.on(cirq.LineQubit(qutrits[0].x),
                                    cirq.LineQubit(qutrits[1].x))
            elif q_type is cirq.NamedQid:
                return base_gate.on(cirq.NamedQubit(qutrits[0].name),
                                    cirq.NamedQubit(qutrits[1].name))

        # If reached, either gate acts on more than 2 qutrits or
        # the type of qid used is not supported
        raise TypeError

    batch_replace = []
    for i, op in circuit.findall_operations(lambda x: True):
        batch_replace.append((i, op, unwrap_gate(op)))
    transformed_circuit = circuit.unfreeze(copy=True)
    transformed_circuit.batch_replace(batch_replace)
    return transformed_circuit
