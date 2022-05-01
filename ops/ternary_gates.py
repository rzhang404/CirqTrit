import cirq
import numpy as np


# with example from https://github.com/quantumlib/Cirq/blob/master/docs/qudits.ipynb
class QutritPlusGate(cirq.Gate):
    """A one-qutrit gate that implements the following action:

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


class QutritMinusGate(cirq.Gate):
    """A one-qutrit gate that implements the following action:

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
    """A two-qutrit gate that implements the following action:

        |ab> -> |ba>
    """

    def _qid_shape_(self):
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
