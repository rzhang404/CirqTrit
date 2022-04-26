import cirq
import numpy as np
from typing import Tuple


# with example from https://github.com/quantumlib/Cirq/blob/master/docs/qudits.ipynb
class QutritPlusGate(cirq.SingleQubitGate):
    """
    |n> -> |n+1 % 3>
    """
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])

    def _circuit_diagram_info_(self, args):
        return '[+1]'

OneControlledPlusGate = cirq.ControlledGate(sub_gate=QutritPlusGate,
                                            num_controls=1,
                                            control_values=[1],
                                            control_qid_shape=(3,))
TwoControlledPlusGate = cirq.ControlledGate(sub_gate=QutritPlusGate,
                                            num_controls=1,
                                            control_values=[2],
                                            control_qid_shape=(3,))

class QutritMinusGate(cirq.SingleQubitGate):
    """
    |n> -> |n-1 % 3>
    """
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]])

    def _circuit_diagram_info_(self, args):
        return '[-1]'


OneControlledMinusGate = cirq.ControlledGate(sub_gate=QutritMinusGate,
                                             num_controls=1,
                                             control_values=[1],
                                             control_qid_shape=(3,))
TwoControlledMinusGate = cirq.ControlledGate(sub_gate=QutritMinusGate,
                                             num_controls=1,
                                             control_values=[2],
                                             control_qid_shape=(3,))

class QutritSwap(cirq.Gate):

    def _qid_shape_(self):
        # I think it's (3,3,), not (9,) or ((3,),(3,)), but needs testing; backed up by
        # https://github.com/quantumlib/Cirq/blob/v0.14.0/cirq-core/cirq/ops/matrix_gates.py#L84
        return (3, 3,)

    def _unitary_(self):
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]])

# Not used, in favour of SingleQubitGateToQutritGate
class TwoStateGate(cirq.SingleQubitGate):
    def __init__(self, gate, states):
        """

        :param gate: base single qubit gate to expand into ternary
        :param states: which two states these are to operate on
        example: TwoStateGate(cirq.X(line_qid[0]), [0,1]) has unitary of
            np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
        TwoStateGate(cirq.Y(line_qid[1]), [0,2]) has
            np.array([[0, 0, -j],
                      [0, 1, 0],
                      [j, 0, 0]])
        """
        self.gate = gate
        self.states = states

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        if self.states != [0, 1]:
            return NotImplementedError

        u = self.gate.unitary
        u = np.vstack([u, [0, 0]])
        u = np.column_stack([u, [0, 0, 1]])
        return u

    def _circuit_diagram_info(self, args):
        return str(self.gate) + str(self.states)



class SingleQubitGateToQutritGate(cirq.Gate):
    """
    Wraps a Cirq simulator gate operating on a single qubit in
    one operating on the first two levels of a qutrit,
    and acting as the identity on |2>.

    Note: not the same as generalizing gates to ternary.
    """

    def __init__(self, gate):
        assert cirq.num_qubits(gate) == 1
        qb_unitary = cirq.unitary(gate)

        # Add one row and one column, filled with zeroes
        qt_unitary = np.pad(qb_unitary, ((0, 1), (0, 1)))

        # No effect to |2> state
        qt_unitary[2][2] = 1
        self.qt_unitary = qt_unitary

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (3,)

    def _unitary_(self):
        return self.qt_unitary


class TwoQubitGateToQutritGate(cirq.Gate):
    """
    Wraps a Cirq simulator gate operating on two qubits in
    one operating on the first two levels of qutrits,
    and acting as the identity on any state with a qutrit in |2>.
    """

    def __init__(self, gate):
        assert cirq.num_qubits(gate) == 2
        qb_unitary = cirq.unitary(gate)

        qt_unitary = np.eye(9, dtype=np.complex128)  # Default: anything involving 2 state is identity
        qt_unitary[0:2, 0:2] = qb_unitary[0:2, 0:2]  # {|00>, |01>}{<00|, <01|}
        qt_unitary[3:5, 0:2] = qb_unitary[2:4, 0:2]  # {|10>, |11>}{<00|, <01|}
        qt_unitary[0:2, 3:5] = qb_unitary[0:2, 2:4]  # {|00>, |01>}{<10|, <11|}
        qt_unitary[3:5, 3:5] = qb_unitary[2:4, 2:4]  # {|10>, |11>}{<10|, <11|}

        self.qt_unitary = qt_unitary

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (3, 3,)

    def _unitary_(self):
        return self.qt_unitary


if __name__ == '__main__':
    qutrits = cirq.LineQid.range(3, dimension=3)

    X3 = SingleQubitGateToQutritGate(cirq.X)
    CNOT3 = TwoQubitGateToQutritGate(cirq.CNOT)

    circtrit = cirq.Circuit(
        X3(qutrits[0]),
        CNOT3(qutrits[0], qutrits[1])
    )

    non_noisy_simulator = cirq.DensityMatrixSimulator()

    all_to_all_ideal = non_noisy_simulator.simulate(circtrit)

    print(all_to_all_ideal)
