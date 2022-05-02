import cirq
import numpy as np
from typing import Dict, Sequence
from ops.channels import (
    QutritKrausChannel,
    SingleQutritDepolarizingChannel,
    TwoQutritDepolarizingChannel,
)


class HardwareAwareSymmetricNoise(cirq.NoiseModel):
    """
    A noise model parameterized on hardware specific single-qudit and two-qudit gate error rates,
    which are distributed into equal likelihoods of occurring as any Pauli error.
    Note that idle dampening errors are still assumed here to be identical.
    """

    def __init__(
        self,
        single_qutrit_hardware_error_rates,  # type: Dict[cirq.Qid][np.float64]
        two_qutrit_hardware_error_rates,  # type: Dict[cirq.Qid][Dict[cirq.Qid][np.float64]]
        lambda_short=100.0 / 10000.0,
        lambda_long=None,
    ):
        """Initializes the noise model.

        Args:
            single_qutrit_hardware_error_rates: A dict of probabilities
                to incur any single qutrit Pauli error when acting on
                the qutrits they are indexed by.
            two_qutrit_hardware_error_rates: A dict of probabilities
                to incur any two-qutrit Pauli error when acting on
                the qutrits they are indexed by.
            lambda_short: The ratio between single qutrit
                gate duration and coherence time T1.
            lambda_long: The ratio between two-qutrit
                gate duration and coherence time T1.
                Defaults to 3 * lambda_short if not given.
        """

        self.single_qutrit_error_rates = single_qutrit_hardware_error_rates
        self.two_qutrit_error_rates = two_qutrit_hardware_error_rates

        if lambda_long is None:
            lambda_long = lambda_short * 3

        # Decay rate is e^(energy level * dt / T1)
        gamma_1_short = 1 - np.exp(-1 * lambda_short)
        gamma_1_long = 1 - np.exp(-1 * lambda_long)
        gamma_2_short = 1 - np.exp(-2 * lambda_short)
        gamma_2_long = 1 - np.exp(-2 * lambda_long)

        for qa in two_qutrit_hardware_error_rates.keys():
            for qb in two_qutrit_hardware_error_rates[qa].keys():
                if not qa in two_qutrit_hardware_error_rates[qb]:
                    # If asymmetric rates are not given, assume symmetric error rates
                    self.two_qutrit_error_rates[qb][qa] = two_qutrit_hardware_error_rates[qa][qb]

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

        # To simulate gate errors, add a ternary Pauli error as
        # depolarizing noise for each physical gate applied,
        # based on which qutrit(s) it is being applied to
        for op in ops:
            op_dim = len(op.qubits)
            max_op_dim = max(max_op_dim, op_dim)
            if op_dim == 1:
                error_rate = self.single_qutrit_error_rates[op.qubits[0]]
                gate_moment = gate_moment.with_operation(
                    SingleQutritDepolarizingChannel(prob=error_rate).on(*op.qubits)
                )
            elif op_dim == 2:
                error_rate = self.two_qutrit_error_rates[op.qubits[0]][op.qubits[1]]
                gate_moment = gate_moment.with_operations(
                    TwoQutritDepolarizingChannel(prob=error_rate).on(*op.qubits)
                )

        effective_noise_moments.append(gate_moment)

        # As idle errors, apply amplitude dampening over all qubits
        # roughly corresponding to the duration of this moment,
        # where two-qubit gates are assumed to dominate in duration
        if max_op_dim == 1:
            idle_moment = cirq.Moment(self.idle_short(qid) for qid in system_qubits)
            effective_noise_moments.append(idle_moment)
        elif max_op_dim == 2:
            idle_moment = cirq.Moment(self.idle_long(qid) for qid in system_qubits)
            effective_noise_moments.append(idle_moment)

        return effective_noise_moments
