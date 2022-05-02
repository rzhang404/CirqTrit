import cirq
import numpy as np
from ops.pauli_operators import single_qutrit_pauli_operators, two_qutrit_pauli_operators


class QutritMixtureChannel(cirq.Gate):  # Can't inherit from SingleQubitGate
    """
    A channel comprised of a mixture of unitaries over qutrits, and corresponding probabilities.

    Used in place of depolarization.
    """

    def __init__(self, error_weights, errors):
        assert np.isclose(sum(error_weights), 1.0)
        op_shape = errors[0].shape
        for error in errors:
            assert error.shape == op_shape
        self.mixture = tuple(zip(error_weights, errors))
        if op_shape == (3, 3):
            assert len(error_weights) == 9
            self.qid_shape = (3,)  # One-qutrit mixture
        elif op_shape == (9, 9):
            assert len(error_weights) == 81
            self.qid_shape = (3, 3)  # Two-qutrit mixture

    def _qid_shape_(self):
        return self.qid_shape

    def _mixture_(self):
        return self.mixture

    def _circuit_diagram_info_(self, args) -> str:
        return (
            f"QutritMixtureChannel"  # would get too long if we printed out all probabilities here
        )


# Kraus operators over qudits are not supported through KrausChannel interface, see
# https://github.com/quantumlib/Cirq/blob/v0.14.0/cirq-core/cirq/ops/kraus_channel.py#L37
class QutritKrausChannel(cirq.Gate):
    """A channel defined with regards to a complete set
    of Kraus operators over a mode of decay.

    Used in place of amplitude dampening.
    """

    def __init__(self, kraus_ops):
        for op in kraus_ops:
            assert op.shape == (3, 3)
        self.kraus_operators = kraus_ops

    def _qid_shape_(self):
        return (3,)

    def _kraus_(self):
        return self.kraus_operators

    def _circuit_diagram_info_(self, args) -> str:
        return f"QutritKrausChannel"


class SingleQutritDepolarizingChannel(QutritMixtureChannel):
    def __init__(self, prob):
        operator_weights = [1 - prob] + 8 * [prob / 8]
        super().__init__(error_weights=operator_weights, errors=single_qutrit_pauli_operators)


class TwoQutritDepolarizingChannel(QutritMixtureChannel):
    def __init__(self, prob):
        operator_weights = [1 - prob] + 80 * [prob / 80]
        super().__init__(error_weights=operator_weights, errors=two_qutrit_pauli_operators)
