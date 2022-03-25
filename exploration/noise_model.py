import cirq
from typing import Dict, Sequence


class GokhaleNoiseModelOnQubits(cirq.NoiseModel):
    def __init__(self, lambda_1, lambda_2, p1=0.0001/3, p2=0.001/15, t1=1):
        self.idle_short = cirq.ops.AmplitudeDampingChannel(lambda_1)
        self.idle_long = cirq.ops.AmplitudeDampingChannel(lambda_1 * 10)  # ...?
        self.p1 = p1
        self.p2 = p2
        self.t1 = t1

    def noisy_moment(
        self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        if self.is_virtual_moment(moment):
            return moment
        effective_noise_moments = [moment]
        ops = moment.operations
        gate_moment = cirq.Moment()
        max_op_dim = 0
        for op in ops:
            op_dim = len(op.qubits)
            max_op_dim = max(max_op_dim, op.qubits)
            if len(op.qubits) == 1:
                gate_moment = gate_moment.with_operation()
            elif len(op.qubits) == 2:
                gate_moment = gate_moment.with_operation()

        effective_noise_moments.append(gate_moment)

        # Idle errors only consider dampening
        if max_op_dim == 1:
            idle_moment = cirq.Moment(self.idle_short(qubit) for qubit in system_qubits)
            effective_noise_moments.append(idle_moment)
        elif max_op_dim == 2:
            idle_moment = cirq.Moment(self.idle_long(qubit) for qubit in system_qubits)
            effective_noise_moments.append(idle_moment)

        return effective_noise_moments


