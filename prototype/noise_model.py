import cirq
import numpy as np
from typing import Dict, Sequence


class GokhaleNoiseModelOnQubits(cirq.NoiseModel):
    """
    An implementation of Gokahle et al.'s noise model over circuits solely composed of qubit gates.

    This is an exploration into custom definitions of noise models before substituting for qutrit-based channels.
    """
    single_qid_errors = [np.eye(2), cirq.X, cirq.Y, cirq.Z]
    two_qid_errors = [np.kron(np.eye(2), np.eye(2))]

    # Kraus operators over qudits do not appear to be supported
    # https://github.com/quantumlib/Cirq/blob/v0.14.0/cirq-core/cirq/ops/kraus_channel.py#L37

    def __init__(self, lambda_1, lambda_2, p1=0.0001 / 3, p2=0.001 / 15, t1=1):
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
            max_op_dim = max(max_op_dim, op_dim)
            if len(op.qubits) == 1:
                gate_moment = gate_moment.with_operation(
                    cirq.depolarize(p=0.001, n_qubits=1).on(*op.qubits))
            elif len(op.qubits) == 2:
                gate_moment = gate_moment.with_operations(
                    cirq.depolarize(p=0.001, n_qubits=2).on(*op.qubits))

        effective_noise_moments.append(gate_moment)

        # Idle errors only consider dampening
        if max_op_dim == 1:
            idle_moment = cirq.Moment(self.idle_short(qubit) for qubit in system_qubits)
            effective_noise_moments.append(idle_moment)
        elif max_op_dim == 2:
            idle_moment = cirq.Moment(self.idle_long(qubit) for qubit in system_qubits)
            effective_noise_moments.append(idle_moment)

        return effective_noise_moments


class QutritMixtureChannel(cirq.Gate):  # Can't inherit from SingleQubitGate
    """
    A channel comprised of a mixture of unitaries over qutrits, and corresponding probabilities.

    Used in place of depolarization.
    """
    def __init__(self, errors, error_weights):
        assert .999 < sum(error_weights) < 1.001
        self.mixture = tuple(zip(error_weights, errors))
        if errors[0].shape == (3, 3):
            self.shape = (3,)  # One-qutrit mixture
        elif errors[0].shape == (9, 9):
            self.shape = (3, 3)  # Two-qutrit mixture

    def _qid_shape_(self):
        return self.shape

    def _mixture_(self):
        return self.mixture

    def _circuit_diagram_info_(self, args) -> str:
        return f"QutritMixtureChannel"  # would get too long if we printed out all probabilities here


class QutritKrausChannel(cirq.Gate):
    """
    A channel comprised of all the Kraus operators over a mode of decay.

    Used in place of amplitude dampening.
    """
    def __init__(self, krause_ops):
        self.kraus_operators = krause_ops

    def _qid_shape_(self):
        return (3,)  # Idle errors only used on individual qudits

    def _kraus_(self):
        return self.kraus_operators

# Pulled from https://github.com/epiqc/qutrits/blob/4ae948006c4ff3d47ae47e69d05dcd3025485665/cirq/circuits/circuit.py#L1493
X3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.complex128)
Z3 = np.array([[1, 0, 0], [0, np.e ** (np.pi * 1j * -2.0 / 3.0), 0], [0, 0, np.e ** (np.pi * 1j * -4.0 / 3.0)]]
              , dtype=np.complex128)
Y3 = X3 @ Z3
V3 = X3 @ Z3 @ Z3

single_qutrit_kraus_operators = [np.eye(3), Z3, Z3 @ Z3, X3, X3 @ Z3, X3 @ Z3 @ Z3, X3 @ X3, X3 @ X3 @ Z3,
                                 X3 @ X3 @ Z3 @ Z3]
two_qutrit_kraus_operators = [np.kron(np.eye(3), np.eye(3)), np.kron(np.eye(3), Z3), np.kron(np.eye(3), Z3 @ Z3),
                              np.kron(np.eye(3), X3), np.kron(np.eye(3), X3 @ Z3), np.kron(np.eye(3), X3 @ Z3 @ Z3),
                              np.kron(np.eye(3), X3 @ X3), np.kron(np.eye(3), X3 @ X3 @ Z3),
                              np.kron(np.eye(3), X3 @ X3 @ Z3 @ Z3),
                              np.kron(Z3, np.eye(3)), np.kron(Z3, Z3), np.kron(Z3, Z3 @ Z3), np.kron(Z3, X3),
                              np.kron(Z3, X3 @ Z3), np.kron(Z3, X3 @ Z3 @ Z3), np.kron(Z3, X3 @ X3),
                              np.kron(Z3, X3 @ X3 @ Z3), np.kron(Z3, X3 @ X3 @ Z3 @ Z3),
                              np.kron(Z3 @ Z3, np.eye(3)), np.kron(Z3 @ Z3, Z3), np.kron(Z3 @ Z3, Z3 @ Z3),
                              np.kron(Z3 @ Z3, X3), np.kron(Z3 @ Z3, X3 @ Z3), np.kron(Z3 @ Z3, X3 @ Z3 @ Z3),
                              np.kron(Z3 @ Z3, X3 @ X3), np.kron(Z3 @ Z3, X3 @ X3 @ Z3),
                              np.kron(Z3 @ Z3, X3 @ X3 @ Z3 @ Z3),
                              np.kron(X3, np.eye(3)), np.kron(X3, Z3), np.kron(X3, Z3 @ Z3), np.kron(X3, X3),
                              np.kron(X3, X3 @ Z3), np.kron(X3, X3 @ Z3 @ Z3), np.kron(X3, X3 @ X3),
                              np.kron(X3, X3 @ X3 @ Z3), np.kron(X3, X3 @ X3 @ Z3 @ Z3),
                              np.kron(X3 @ Z3, np.eye(3)), np.kron(X3 @ Z3, Z3), np.kron(X3 @ Z3, Z3 @ Z3),
                              np.kron(X3 @ Z3, X3), np.kron(X3 @ Z3, X3 @ Z3), np.kron(X3 @ Z3, X3 @ Z3 @ Z3),
                              np.kron(X3 @ Z3, X3 @ X3), np.kron(X3 @ Z3, X3 @ X3 @ Z3),
                              np.kron(X3 @ Z3, X3 @ X3 @ Z3 @ Z3),
                              np.kron(X3 @ Z3 @ Z3, np.eye(3)), np.kron(X3 @ Z3 @ Z3, Z3),
                              np.kron(X3 @ Z3 @ Z3, Z3 @ Z3), np.kron(X3 @ Z3 @ Z3, X3), np.kron(X3 @ Z3 @ Z3, X3 @ Z3),
                              np.kron(X3 @ Z3 @ Z3, X3 @ Z3 @ Z3), np.kron(X3 @ Z3 @ Z3, X3 @ X3),
                              np.kron(X3 @ Z3 @ Z3, X3 @ X3 @ Z3), np.kron(X3 @ Z3 @ Z3, X3 @ X3 @ Z3 @ Z3),
                              np.kron(X3 @ X3, np.eye(3)), np.kron(X3 @ X3, Z3), np.kron(X3 @ X3, Z3 @ Z3),
                              np.kron(X3 @ X3, X3), np.kron(X3 @ X3, X3 @ Z3), np.kron(X3 @ X3, X3 @ Z3 @ Z3),
                              np.kron(X3 @ X3, X3 @ X3), np.kron(X3 @ X3, X3 @ X3 @ Z3),
                              np.kron(X3 @ X3, X3 @ X3 @ Z3 @ Z3),
                              np.kron(X3 @ X3 @ Z3, np.eye(3)), np.kron(X3 @ X3 @ Z3, Z3),
                              np.kron(X3 @ X3 @ Z3, Z3 @ Z3), np.kron(X3 @ X3 @ Z3, X3), np.kron(X3 @ X3 @ Z3, X3 @ Z3),
                              np.kron(X3 @ X3 @ Z3, X3 @ Z3 @ Z3), np.kron(X3 @ X3 @ Z3, X3 @ X3),
                              np.kron(X3 @ X3 @ Z3, X3 @ X3 @ Z3), np.kron(X3 @ X3 @ Z3, X3 @ X3 @ Z3 @ Z3),
                              np.kron(X3 @ X3 @ Z3 @ Z3, np.eye(3)), np.kron(X3 @ X3 @ Z3 @ Z3, Z3),
                              np.kron(X3 @ X3 @ Z3 @ Z3, Z3 @ Z3), np.kron(X3 @ X3 @ Z3 @ Z3, X3),
                              np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ Z3), np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ Z3 @ Z3),
                              np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ X3), np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ X3 @ Z3),
                              np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ X3 @ Z3 @ Z3)]


class GokhaleNoiseModelOnQutrits(cirq.NoiseModel):
    def __init__(self,
                 single_qutrit_error_weights,
                 two_qutrit_error_weights,
                 # p1=0.0001 / 3, p2=0.001 / 15, t1=1
                 gamma_1_short=1 - np.exp(-1 * 100.0 / 100000.0),
                 gamma_1_long=1 - np.exp(-1 * 300.0 / 100000.0),
                 gamma_2_short=1 - np.exp(-2 * 100.0 / 100000.0),
                 gamma_2_long=1 - np.exp(-2 * 300.0 / 100000.0),
                 ):
        # self.idle_short = cirq.ops.AmplitudeDampingChannel(lambda_1)  # Works only over single qubits
        # self.idle_long = cirq.ops.AmplitudeDampingChannel(lambda_2)
        self.short_idle_channel_operators = [
            np.array([[1, 0, 0], [0, (1 - gamma_1_short) ** .5, 0], [0, 0, (1 - gamma_2_short) ** .5]]),
            np.array([[0, gamma_1_short ** 0.5, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, gamma_2_short ** 0.5], [0, 0, 0], [0, 0, 0]])]
        self.long_idle_channel_operators = [
            np.array([[1, 0, 0], [0, (1 - gamma_1_long) ** .5, 0], [0, 0, (1 - gamma_2_long) ** .5]]),
            np.array([[0, gamma_1_long ** 0.5, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, gamma_2_long ** 0.5], [0, 0, 0], [0, 0, 0]])]
        # self.idle_short = cirq.KrausChannel(self.short_idle_channel_operators)  # Also only works over qubits
        # self.idle_long = cirq.KrausChannel(self.long_idle_channel_operators)
        self.idle_short = QutritKrausChannel(self.short_idle_channel_operators)
        self.idle_long = QutritKrausChannel(self.long_idle_channel_operators)
        self.single_qutrit_error_weights = single_qutrit_error_weights
        self.two_qutrit_error_weights = two_qutrit_error_weights
        # self.p1 = p1
        # self.p2 = p2
        # self.t1 = t1

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
            op_dim = len(op.qubits)  # Still named qubits for accessing qid shape
            max_op_dim = max(max_op_dim, op_dim)
            if op_dim == 1:
                # weighted_draw(self.single_qutrit_error_weights) # only works for stochastic, not full density sim
                gate_moment = gate_moment.with_operation(
                    QutritMixtureChannel(error_weights=self.single_qutrit_error_weights,
                                         errors=single_qutrit_kraus_operators).on(*op.qubits))
            elif op_dim == 2:
                gate_moment = gate_moment.with_operations(
                    QutritMixtureChannel(error_weights=self.two_qutrit_error_weights,
                                         errors=two_qutrit_kraus_operators).on(*op.qubits))

        effective_noise_moments.append(gate_moment)

        # Idle errors dampen over time, always, as Kraus operators
        if max_op_dim == 1:
            idle_moment = cirq.Moment(self.idle_short(qid) for qid in system_qubits)
            effective_noise_moments.append(idle_moment)
        elif max_op_dim == 2:
            idle_moment = cirq.Moment(self.idle_long(qid) for qid in system_qubits)
            effective_noise_moments.append(idle_moment)

        return effective_noise_moments


class SingleQutritDampeningChannel(QutritMixtureChannel):
    def __init__(self, prob):
        single_qutrit_kraus_operator_weights = [1 - 8 * prob] + 8 * [prob]
        super().__init__(single_qutrit_kraus_operators, single_qutrit_kraus_operator_weights)


class HardwareAwareSymmetricNoise(cirq.NoiseModel):
    """
    A noise model parameterized on hardware specific single-qudit and two-qudit gate error rates,
    which are distributed into equal likelihoods of occurring as any Pauli error.
    Note that idle errors are assumed to be identical here.
    """

    def __init__(self,
                 single_qutrit_error_rates,  # type: Dict[cirq.Qid][np.float64]
                 two_qutrit_error_rates,  # type: Dict[cirq.Qid][Dict[cirq.Qid][np.float64]]
                 gamma_1_short=1 - np.exp(-1 * 100.0 / 100000.0),
                 gamma_1_long=1 - np.exp(-1 * 300.0 / 100000.0),
                 gamma_2_short=1 - np.exp(-2 * 100.0 / 100000.0),
                 gamma_2_long=1 - np.exp(-2 * 300.0 / 100000.0),
                 ):


        self.single_qutrit_error_rates = single_qutrit_error_rates
        self.two_qutrit_error_rates = two_qutrit_error_rates
        for qa in two_qutrit_error_rates.keys():
            for qb in two_qutrit_error_rates[qa].keys():
                if not qa in two_qutrit_error_rates[qb]:
                    # If asymmetric rates are not given, use symmetric CNOT error rates
                    self.two_qutrit_error_rates[qb][qa] = two_qutrit_error_rates[qa][qb]
        self.short_idle_channel_operators = [
            np.array([[1, 0, 0], [0, (1 - gamma_1_short) ** .5, 0], [0, 0, (1 - gamma_2_short) ** .5]]),
            np.array([[0, gamma_1_short ** 0.5, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, gamma_2_short ** 0.5], [0, 0, 0], [0, 0, 0]])]
        self.long_idle_channel_operators = [
            np.array([[1, 0, 0], [0, (1 - gamma_1_long) ** .5, 0], [0, 0, (1 - gamma_2_long) ** .5]]),
            np.array([[0, gamma_1_long ** 0.5, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, gamma_2_long ** 0.5], [0, 0, 0], [0, 0, 0]])]
        self.idle_short = QutritKrausChannel(self.short_idle_channel_operators)
        self.idle_long = QutritKrausChannel(self.long_idle_channel_operators)

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
            max_op_dim = max(max_op_dim, op_dim)
            if op_dim == 1:
                p_1 = self.single_qutrit_error_rates[op.qubits[0]]
                error_weights = [1 - 8 * p_1] + 8 * [p_1]
                gate_moment = gate_moment.with_operation(
                    QutritMixtureChannel(error_weights=error_weights,
                                         errors=single_qutrit_kraus_operators).on(*op.qubits))
            elif op_dim == 2:
                p_2 = self.two_qutrit_error_rates[op.qubits[0]][op.qubits[1]]
                error_weights = [1 - 80 * p_2] + 80 * [p_2]
                gate_moment = gate_moment.with_operations(
                    QutritMixtureChannel(error_weights=error_weights,
                                         errors=two_qutrit_kraus_operators).on(*op.qubits))

        effective_noise_moments.append(gate_moment)

        # Idle errors dampen over time, always, as Kraus operators
        if max_op_dim == 1:
            idle_moment = cirq.Moment(self.idle_short(qid) for qid in system_qubits)
            effective_noise_moments.append(idle_moment)
        elif max_op_dim == 2:
            idle_moment = cirq.Moment(self.idle_long(qid) for qid in system_qubits)
            effective_noise_moments.append(idle_moment)

        return effective_noise_moments

