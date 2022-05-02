import cirq
import pytket

# from pytket.extensions.cirq import cirq_to_tk, tk_to_cirq  # interesting, investigate .cirq.cirq_conversion package for later
from pytket_cirq_extension.cirq_to_tket import cirq_to_tk
from pytket_cirq_extension.tket_to_cirq import tk_to_cirq

# from pytket.architecture  # cannot import architectures directly
from pytket.predicates import CompilationUnit, ConnectivityPredicate
from pytket.passes import PlacementPass, RoutingPass
from pytket.placement import GraphPlacement
from pytket.transform import Transform

# Make unconstrained input circuit
num_input_qubits = 10
circuit_depth = 6
op_density = 0.8  # probability at each moment a qubit is to have a gate acting on it
gate_domain = {  # cirq.X: 1,
    # cirq.Y: 1,
    # cirq.Z: 1,
    # cirq.H: 1,
    # cirq.S: 1,
    # cirq.T: 1,
    cirq.CNOT: 2,
    cirq.CZ: 2,
    # cirq.SWAP: 2,
    cirq.ISWAP: 2,
    # cirq.CZPowGate: 2,  # Somehow causing problems rn
    # cirq.TOFFOLI: 3,  # Not supported by tket
}
in_circ = cirq.testing.random_circuit(
    qubits=num_input_qubits, n_moments=circuit_depth, op_density=op_density, gate_domain=gate_domain
)

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

print(placed_circ)

route = RoutingPass(tk_dev)
route.apply(comp_unit)

routed_tk_circ = comp_unit.circuit
Transform.DecomposeBRIDGE().apply(routed_tk_circ)

out_circ = tk_to_cirq(comp_unit.circuit)

print(out_circ)

reps = 1000
noise_model = cirq.generalized_amplitude_damp(p=0.001, gamma=0.01)
# noise_model = GokhaleNoiseModelOnQubits(lambda_1=0.0001, lambda_2=0.001)

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
