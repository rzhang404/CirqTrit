import cirq
import pytket
from pytket.extensions.cirq import cirq_to_tk, tk_to_cirq  # interesting, investigate .cirq.cirq_conversion package for later
# from pytket.architecture  # cannot import architectures directly
from pytket.predicates import CompilationUnit, ConnectivityPredicate
from pytket.passes import PlacementPass, RoutingPass, SequencePass
from pytket.routing import GraphPlacement
from cirq.experiments import linear_xeb_fidelity

# Make unconstrained input circuit
num_input_qubits = 10
circuit_depth = 10
op_density = 0.5  # probability at each moment a qubit is to have a gate acting on it
gate_domain = {cirq.X: 1,
               cirq.Y: 1,
               cirq.Z: 1,
               cirq.H: 1,
               cirq.S: 1,
               cirq.T: 1,
               cirq.CNOT: 2,
               cirq.CZ: 2,
               cirq.SWAP: 2,
               cirq.ISWAP: 2,
               # cirq.CZPowGate: 2,  # Somehow causing problems rn
               # cirq.TOFFOLI: 3,  # Not supported by tket
               }
in_circ = cirq.testing.random_circuit(qubits=num_input_qubits,
                                      n_moments=circuit_depth,
                                      op_density=op_density,
                                      gate_domain=gate_domain)


# TODO: make constrained placed/routed circuit
tk_circ = pytket.extensions.cirq.cirq_to_tk(in_circ)
# tk_dev = pytket.extensions.cirq.CirqDensityMatrixSimBackend()  # args?
tk_dev = pytket.routing.SquareGrid()
comp_unit = CompilationUnit(tk_circ, [ConnectivityPredicate(tk_dev)])
place_and_route = SequencePass([
    PlacementPass(GraphPlacement(tk_dev)),
    RoutingPass(tk_dev)])
place_and_route.apply(comp_unit)

out_circ = tk_to_cirq(comp_unit.final_map)

reps = 1000
noise_model = cirq.generalized_amplitude_damp(p=0.1, gamma=0.5)  # TODO: find how to add multiple channels
non_noisy_simulator = cirq.DensityMatrixSimulator()
noisy_simulator = cirq.DensityMatrixSimulator(noise=noise_model)

all_to_all_ideal = non_noisy_simulator.simulate(in_circ)
all_to_all_distribution = non_noisy_simulator.run(in_circ, repetitions=reps)

all_to_all_noisy_ideal = noisy_simulator.simulate(in_circ)
all_to_all_noisy_distribution = noisy_simulator.run(in_circ, repetitions=reps)

routed_ideal = non_noisy_simulator.simulate(out_circ)
routed_distribution = non_noisy_simulator.run(out_circ, repetitions=reps)

routed_noisy = noisy_simulator.simulate(out_circ)
routed_noisy_distribution = noisy_simulator.run(out_circ, repetitions=reps)

# linear_xeb_fidelity(circuit=routed_noisy)