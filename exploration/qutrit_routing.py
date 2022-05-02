import cirq
import pytket
from pytket_cirq_extension.cirq_to_tket import cirq_to_tk
from pytket_cirq_extension.tket_to_cirq import tk_to_cirq

# from pytket.architecture  # cannot import architectures directly
from pytket.predicates import CompilationUnit, ConnectivityPredicate
from pytket.passes import PlacementPass, RoutingPass
from pytket.placement import GraphPlacement
from pytket.transform import Transform
from ops.to_qutrit_wrappers import (
    SingleQubitGateToQutritGate,
    TwoQubitGateToQutritGate,
)
from ops.ternary_gates import (
    QutritPlusGate,
    OneControlledPlusGate,
    TwoControlledPlusGate,
    QutritMinusGate,
    OneControlledMinusGate,
    TwoControlledMinusGate,
)
from transformations.dimension_transform import qutrit_to_qubit, qubit_to_qutrit
from noise_models.gokhale_qutrit import GokhaleNoiseModelOnQutrits


# Make unconstrained input circuit
input_qutrits = [cirq.NamedQid(str(i), dimension=3) for i in range(10)]
circuit_depth = 4
op_density = 0.3  # probability at each moment a qubit is to have a gate acting on it
gate_domain = {
    SingleQubitGateToQutritGate(cirq.X): 1,
    SingleQubitGateToQutritGate(cirq.Y): 1,
    SingleQubitGateToQutritGate(cirq.Z): 1,
    SingleQubitGateToQutritGate(cirq.H): 1,
    SingleQubitGateToQutritGate(cirq.S): 1,
    SingleQubitGateToQutritGate(cirq.T): 1,
    TwoQubitGateToQutritGate(cirq.CNOT): 2,
    TwoQubitGateToQutritGate(cirq.CZ): 2,
    TwoQubitGateToQutritGate(cirq.SWAP): 2,
    TwoQubitGateToQutritGate(cirq.ISWAP): 2,
    # QutritPlusGate: 1,
    # OneControlledPlusGate: 2,
    # TwoControlledPlusGate: 2,
    # QutritMinusGate: 1,
    # OneControlledMinusGate: 2,
    # TwoControlledMinusGate: 2,
    # cirq.CZPowGate: 2,  # Somehow causing problems rn
    # cirq.TOFFOLI: 3,  # Not supported by tket
}
in_circ = cirq.testing.random_circuit(
    qubits=input_qutrits, n_moments=circuit_depth, op_density=op_density, gate_domain=gate_domain
)
in_circ = qutrit_to_qubit(in_circ)
in_circ = cirq.transformers.align_left(in_circ)
tk_circ = cirq_to_tk(in_circ)
check_circ = tk_to_cirq(tk_circ)
check_circ = cirq.transformers.align_left(check_circ)
assert in_circ == check_circ

tk_dev = pytket.architecture.SquareGrid(2, 5)
comp_unit = CompilationUnit(tk_circ, [ConnectivityPredicate(tk_dev)])
place = PlacementPass(GraphPlacement(tk_dev))
place.apply(comp_unit)

placed_circ = tk_to_cirq(comp_unit.circuit)

placed_circ = qubit_to_qutrit(placed_circ)
print(placed_circ)

route = RoutingPass(tk_dev)
route.apply(comp_unit)

routed_tk_circ = comp_unit.circuit
Transform.DecomposeBRIDGE().apply(routed_tk_circ)

out_circ = tk_to_cirq(comp_unit.circuit)
out_circ = qubit_to_qutrit(out_circ)

print(out_circ)

reps = 1000
p_1 = 0.001 / 3
p_2 = 0.01 / 15
noise_model = GokhaleNoiseModelOnQutrits(
    single_qutrit_error_weights=[1 - p_1] + 8 * [p_1 / 8],
    two_qutrit_error_weights=[1 - p_2] + 80 * [p_2 / 80],
    lambda_short=100.0 / 10000.0,
    lambda_long=300.0 / 10000.0,
)

non_noisy_simulator = cirq.DensityMatrixSimulator()
noisy_simulator = cirq.DensityMatrixSimulator(noise=noise_model)


all_to_all_ideal = non_noisy_simulator.simulate(placed_circ)
# all_to_all_distribution = non_noisy_simulator.run(in_circ, repetitions=reps)

all_to_all_noisy_ideal = noisy_simulator.simulate(placed_circ)
# all_to_all_noisy_distribution = noisy_simulator.run(in_circ, repetitions=reps)

routed_ideal = non_noisy_simulator.simulate(out_circ)
# routed_distribution = non_noisy_simulator.run(out_circ, repetitions=reps)

routed_noisy = noisy_simulator.simulate(out_circ)
# routed_noisy_distribution = noisy_simulator.run(out_circ, repetitions=reps)


print(
    cirq.qis.fidelity(
        all_to_all_ideal.final_density_matrix,
        all_to_all_ideal.final_density_matrix,
        (len(all_to_all_ideal.final_density_matrix),),
    )
)

print(
    cirq.qis.fidelity(
        all_to_all_ideal.final_density_matrix,
        all_to_all_noisy_ideal.final_density_matrix,
        (len(all_to_all_ideal.final_density_matrix),),
    )
)


print(
    cirq.qis.fidelity(
        all_to_all_ideal.final_density_matrix,
        routed_ideal.final_density_matrix,
        (len(all_to_all_ideal.final_density_matrix),),
    )
)

print(
    cirq.qis.fidelity(
        all_to_all_ideal.final_density_matrix,
        routed_noisy.final_density_matrix,
        (len(all_to_all_ideal.final_density_matrix),),
    )
)
