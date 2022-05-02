import cirq
import numpy as np
from typing import Sequence


class GokhaleNoiseModelOnQubits(cirq.NoiseModel):
    """An implementation of Gokahle et al.'s noise model over
    circuits solely composed of qubit gates.

    This is an exploration into custom definitions of noise models
    before substituting for qutrit-based channels.
    """

    def __init__(
        self,
        lambda_short=100.0 / 10000.0,
        lambda_long=None,
        p1=0.0001 / 3,
        p2=0.001 / 15,
    ):
        """Initializes the noise model.

        Args:
            lambda_short: The ratio between single-qubit
                gate duration and coherence time T1.
            lambda_long: The ratio between two-qubit
                gate duration and coherence time T1.
                Defaults to 3 * lambda_short if not given.
            p1: The probability a single-qubit gate incurs a Pauli error.
            p2: The probability a two-qubit gate incurs a Pauli error.
        """
        # If not provided, two-qubit gates are assumed to be
        # three times the duration of single-qubit gates,
        # see https://arxiv.org/pdf/1702.01852.pdf
        if lambda_long is None:
            lambda_long = lambda_short * 3

        gamma_short = 1 - np.exp(-1 * lambda_short)
        gamma_long = 1 - np.exp(-1 * lambda_long)
        self.idle_short = cirq.ops.AmplitudeDampingChannel(gamma_short)
        self.idle_long = cirq.ops.AmplitudeDampingChannel(gamma_long)
        self.p1 = p1
        self.p2 = p2

    def noisy_moment(
        self, moment: "cirq.Moment", system_qubits: Sequence["cirq.Qid"]
    ) -> "cirq.OP_TREE":
        """Substitutes a noiseless moment with an equivalent one with noise added.

        Args:
            moment: An input moment to apply noise to.
            system_qubits: The full list of qudits in the system.
        Returns:
            A collection of moments equivalent to noisily executing the input.
        """

        # Apply no noise if nothing is physically happening
        if self.is_virtual_moment(moment):
            return moment

        effective_noise_moments = [moment]
        ops = moment.operations
        gate_moment = cirq.Moment()
        max_op_dim = 0

        # To simulate gate errors, add a probabilistic Pauli error
        # as depolarizing noise for each physical gate applied
        for op in ops:
            op_dim = len(op.qubits)
            max_op_dim = max(max_op_dim, op_dim)
            if len(op.qubits) == 1:
                gate_moment = gate_moment.with_operation(
                    cirq.depolarize(p=0.001, n_qubits=1).on(*op.qubits)
                )
            elif len(op.qubits) == 2:
                gate_moment = gate_moment.with_operations(
                    cirq.depolarize(p=0.001, n_qubits=2).on(*op.qubits)
                )

        effective_noise_moments.append(gate_moment)

        # As idle errors, apply amplitude dampening over all qubits
        # roughly corresponding to the duration of this moment,
        # where two-qubit gates are assumed to dominate in duration
        if max_op_dim == 1:
            idle_moment = cirq.Moment(self.idle_short(qubit) for qubit in system_qubits)
            effective_noise_moments.append(idle_moment)
        elif max_op_dim == 2:
            idle_moment = cirq.Moment(self.idle_long(qubit) for qubit in system_qubits)
            effective_noise_moments.append(idle_moment)

        return effective_noise_moments
