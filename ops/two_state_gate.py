import cirq
import numpy as np


# Not used, abandoned in favour of SingleQubitGateToQutritGate,
# as Kraus operators no longer need to be generated.
class TwoStateGate(cirq.SingleQubitGate):
    def __init__(self, gate, states):
        """A one-qutrit gate that embeds a one-qubit gate in a subspace

        Args:
            gate: The base single qubit gate to expand into ternary.
            states: The two states these are to operate on as a subspace.

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
