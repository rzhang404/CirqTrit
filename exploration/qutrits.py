import cirq
import numpy as np

# with example from https://github.com/quantumlib/Cirq/blob/master/docs/qudits.ipynb
class QutritPlusGate(cirq.SingleQubitGate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])

    def _circuit_diagram_info_(self, args):
        return '[+1]'


class QutritMinusGate(cirq.SingleQubitGate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]])

    def _circuit_diagram_info_(self, args):
        return '[-1]'


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
        if self.states != [0,1]:
            return NotImplementedError

        u = self.gate.unitary
        u = np.vstack([u, [0,0]])
        u = np.column_stack([u, [0, 0, 1]])
        return u

    def _circuit_diagram_info(self, args):
        return str(self.gate) + str(self.states)


class QutritSwap(cirq.Gate):

    def _qid_shape_(self):
        return (3,3,)  #?

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


# Qid syntax allows dimension to be specified for qudits
qutrits = cirq.LineQid.range(3, dimension=3)

circtrit = cirq.Circuit(
)
