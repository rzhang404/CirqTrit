from typing import Sequence

import cirq
import numpy as np

from ops.channels import QutritKrausChannel, QutritMixtureChannel
from ops.pauli_operators import (
    single_qutrit_pauli_operators,
    two_qutrit_pauli_operators,
)


class GokhaleNoiseModelOnQutrits(cirq.NoiseModel):
    """A replication of the noise model used in
    https://dl.acm.org/doi/pdf/10.1145/3307650.3322253,
    parameterized on gate error weights for each operator,
    and independent of where the qutrits being acted on are.

    Idle errors are with respect to the ratio between
    gate duration and coherence time.
    """

    def __init__(
        self,
        single_qutrit_error_weights: Sequence[float],
        two_qutrit_error_weights: Sequence[float],
        lambda_short=100.0 / 10000.0,
        lambda_long=None,
    ):
        """Initializes noise model.

        Args:
            single_qutrit_error_weights: A list of probabilities
                to incur single qutrit Pauli errors errors when
                applying an operation.
            two_qutrit_error_weights: A list of probabilities
                to incur specific two-qutrit Pauli errors when
                applying an operation.
            lambda_short: The ratio between single qutrit
                gate duration and coherence time T1.
            lambda_long: The ratio between two-qutrit
                gate duration and coherence time T1.
                Defaults to 3 * lambda_short if not given.
        """
        self.single_qutrit_error_weights = single_qutrit_error_weights
        self.two_qutrit_error_weights = two_qutrit_error_weights

        if lambda_long is None:
            lambda_long = lambda_short * 3

        # Decay rate is e^(energy level * dt / T1)
        gamma_1_short = 1 - np.exp(-1 * lambda_short)
        gamma_1_long = 1 - np.exp(-1 * lambda_long)
        gamma_2_short = 1 - np.exp(-2 * lambda_short)
        gamma_2_long = 1 - np.exp(-2 * lambda_long)

        # Assumption: decay is directly from |1> to |0> and from |2> to |0>,
        # but decay from |2> to |1> is negligible
        short_idle_channel_operators = [
            np.array(
                [
                    [1, 0, 0],
                    [0, (1 - gamma_1_short) ** 0.5, 0],
                    [0, 0, (1 - gamma_2_short) ** 0.5],
                ]
            ),
            np.array([[0, gamma_1_short**0.5, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, gamma_2_short**0.5], [0, 0, 0], [0, 0, 0]]),
        ]
        long_idle_channel_operators = [
            np.array(
                [
                    [1, 0, 0],
                    [0, (1 - gamma_1_long) ** 0.5, 0],
                    [0, 0, (1 - gamma_2_long) ** 0.5],
                ]
            ),
            np.array([[0, gamma_1_long**0.5, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, gamma_2_long**0.5], [0, 0, 0], [0, 0, 0]]),
        ]

        self.idle_short = QutritKrausChannel(short_idle_channel_operators)
        self.idle_long = QutritKrausChannel(long_idle_channel_operators)

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
            if op_dim == 1:
                gate_moment = gate_moment.with_operation(
                    QutritMixtureChannel(
                        error_weights=self.single_qutrit_error_weights,
                        errors=single_qutrit_pauli_operators,
                    ).on(*op.qubits)
                )
            elif op_dim == 2:
                gate_moment = gate_moment.with_operations(
                    QutritMixtureChannel(
                        error_weights=self.two_qutrit_error_weights,
                        errors=two_qutrit_pauli_operators,
                    ).on(*op.qubits)
                )

        effective_noise_moments.append(gate_moment)

        # As idle errors, apply amplitude dampening over all qubits
        # roughly corresponding to the duration of this moment, where
        # the presence of two-qubit gates are assumed to dominate in duration
        if max_op_dim == 1:
            idle_moment = cirq.Moment(self.idle_short(qid) for qid in system_qubits)
            effective_noise_moments.append(idle_moment)
        elif max_op_dim == 2:
            idle_moment = cirq.Moment(self.idle_long(qid) for qid in system_qubits)
            effective_noise_moments.append(idle_moment)

        return effective_noise_moments
