import cirq
import pytket
from pytket_cirq_extension.cirq_to_tket import cirq_to_tk
from pytket_cirq_extension.tket_to_cirq import tk_to_cirq
from pytket.predicates import CompilationUnit, ConnectivityPredicate
from pytket.passes import PlacementPass, RoutingPass
from pytket.placement import GraphPlacement
from pytket.transform import Transform
from transformations.dimension_transform import qutrit_to_qubit, qubit_to_qutrit


def place_and_route(circuit: cirq.Circuit, architecture: pytket.architecture.Architecture):
    """Given an abstract circuit and connectivity constraints,
    place all qubits and route them to compile
    an equivalent circuit obeying those constraints.

    Args:
        circuit: The circuit to be compiled.
        architecture: A device representing the
         connectivity constraints to follow.
    """
    in_circ = qutrit_to_qubit(circuit)
    tk_circ = cirq_to_tk(in_circ)

    comp_unit = CompilationUnit(tk_circ, [ConnectivityPredicate(architecture)])
    place = PlacementPass(GraphPlacement(architecture))
    place.apply(comp_unit)

    placed_circ = tk_to_cirq(comp_unit.circuit)
    placed_circ = qubit_to_qutrit(placed_circ)

    route = RoutingPass(architecture)
    route.apply(comp_unit)

    # Some BRIDGEs are included during compilation, which must be decomposed.
    # It is not verified whether this affects any possible ternary states in decomposition.
    routed_tk_circ = comp_unit.circuit
    Transform.DecomposeBRIDGE().apply(routed_tk_circ)

    out_circ = tk_to_cirq(comp_unit.circuit)
    out_circ = qubit_to_qutrit(out_circ)

    return placed_circ, out_circ
