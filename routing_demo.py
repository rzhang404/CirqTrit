import cirq
import pytket

from ops.to_qutrit_wrappers import (
    SingleQubitGateToQutritGate,
    TwoQubitGateToQutritGate,
)
from noise_models.gokhale_qutrit import GokhaleNoiseModelOnQutrits
from transformations.pytket_transforms import place_and_route


# Make unconstrained input circuit
input_qutrits = [cirq.NamedQid(str(i), dimension=3) for i in range(8)]
circuit_depth = 6
op_density = 0.3  # probability at each moment a qubit is to have a gate acting on it
routable_gate_domain = {
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
    qubits=input_qutrits,
    n_moments=circuit_depth,
    op_density=op_density,
    gate_domain=routable_gate_domain,
)
print("Generated circuit:", in_circ)

tk_dev = pytket.architecture.SquareGrid(3, 4)
placed_circ, out_circ = place_and_route(in_circ, tk_dev)

print("Placed circuit:", placed_circ)

print("Routed circuit:", out_circ)

p_1 = 0.001 / 3
p_2 = 0.01 / 15
noise_model = GokhaleNoiseModelOnQutrits(
    single_qutrit_error_weights=[1 - p_1] + 8 * [p_1 / 8],
    two_qutrit_error_weights=[1 - p_2] + 80 * [p_2 / 80],
    lambda_short=100.0 / 10000.0,
    lambda_long=300.0 / 10000.0,
)

ideal_simulator = cirq.DensityMatrixSimulator()
noisy_simulator = cirq.DensityMatrixSimulator(noise=noise_model)

all_to_all_ideal = ideal_simulator.simulate(placed_circ)

routed_noisy = noisy_simulator.simulate(out_circ)

print(
    "Resulting fidelity: ",
    cirq.qis.fidelity(
        all_to_all_ideal.final_density_matrix,
        routed_noisy.final_density_matrix,
        (len(all_to_all_ideal.final_density_matrix),),
    ),
)
