import cirq
import numpy as np
from typing import Tuple


class SingleQubitGateToQutritGate(cirq.Gate):
    """Wraps a single-qubit gate in a single-qutrit gate
    that applies the same action on the first two levels
    of a qutrit, and acts as the identity on |2>.

    Note: not the same as generalizing gates to ternary.
    """

    def __init__(self, gate):
        self.base_gate = gate
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

    def _circuit_diagram_info_(self, args):
        return "{}(01)".format(str(self.base_gate))


class TwoQubitGateToQutritGate(cirq.Gate):
    """Wraps a two-qubit gate in a two-qutrit gate
    that applies the same action on the first two levels
    of two qutrits, and acts as the identity on any state
    with at least one qutrit in |2>.

    Note: not the same as generalizing gates to ternary.
    """

    def __init__(self, gate):
        self.base_gate = gate
        assert cirq.num_qubits(gate) == 2
        qb_unitary = cirq.unitary(gate)

        qt_unitary = np.eye(
            9, dtype=np.complex128
        )  # Default: anything involving 2 state is identity
        qt_unitary[0:2, 0:2] = qb_unitary[0:2, 0:2]  # {|00>, |01>}{<00|, <01|}
        qt_unitary[3:5, 0:2] = qb_unitary[2:4, 0:2]  # {|10>, |11>}{<00|, <01|}
        qt_unitary[0:2, 3:5] = qb_unitary[0:2, 2:4]  # {|00>, |01>}{<10|, <11|}
        qt_unitary[3:5, 3:5] = qb_unitary[2:4, 2:4]  # {|10>, |11>}{<10|, <11|}

        self.qt_unitary = qt_unitary

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (
            3,
            3,
        )

    def _unitary_(self):
        return self.qt_unitary

    def _circuit_diagram_info_(self, args):
        def wrap_wire_symbol(symbol):
            if symbol == "@":
                return "@(1)"  # Controlled on |1>
            else:
                return "{}(01)".format(symbol)

        qubit_diagram_info = cirq.circuit_diagram_info(self.base_gate)
        new_wire_symbols = (wrap_wire_symbol(symbol) for symbol in qubit_diagram_info.wire_symbols)
        return qubit_diagram_info.with_wire_symbols(new_wire_symbols)


if __name__ == "__main__":
    qutrits = cirq.LineQid.range(3, dimension=3)

    X3 = SingleQubitGateToQutritGate(cirq.X)
    CNOT3 = TwoQubitGateToQutritGate(cirq.CNOT)

    circtrit = cirq.Circuit(X3(qutrits[0]), CNOT3(qutrits[0], qutrits[1]))

    non_noisy_simulator = cirq.DensityMatrixSimulator()

    all_to_all_ideal = non_noisy_simulator.simulate(circtrit)

    print(all_to_all_ideal)
