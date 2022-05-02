import cirq
import numpy as np
from typing import Tuple


class SingleQutritGateToQubitGate(cirq.Gate):
    """Wraps a single-qutrit gate in a single-qubit gate
    with dummy action, for the sole purposes of getting to routing.

    Should not be executed in principle.
    """

    def __init__(self, gate):
        self.base_gate = gate

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,)

    def _unitary_(self):
        # Dummy action: Identity
        return np.eye(2, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return "{}(3)".format(str(self.base_gate))


class TwoQutritGateToQubitGate(cirq.Gate):
    """ "Wraps a single-qutrit gate in a single-qubit gate
    with dummy action, for the sole purposes of getting to routing.

    Should not be executed in principle.
    """

    def __init__(self, gate):
        self.base_gate = gate

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (
            3,
            3,
        )

    def _unitary_(self):
        # Dummy action: Identity
        return np.eye(4, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        def wrap_wire_symbol(symbol):
            if symbol == "@":
                return "@(3)"  # Ambiguous but still hint as to invalid gate
            else:
                return "{}(3)".format(symbol)

        qubit_diagram_info = cirq.circuit_diagram_info(self.base_gate)
        new_wire_symbols = (wrap_wire_symbol(symbol) for symbol in qubit_diagram_info.wire_symbols)
        return qubit_diagram_info.with_wire_symbols(new_wire_symbols)
