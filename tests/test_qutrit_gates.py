import cirq
import numpy as np
from ops.to_qutrit_wrappers import SingleQubitGateToQutritGate, TwoQubitGateToQutritGate
from noise_models.hardware_aware import HardwareAwareSymmetricNoise
from noise_models.gokhale_qutrit import GokhaleNoiseModelOnQutrits
import pytest


@pytest.fixture
def args():
    qutrits = cirq.LineQid.range(3, dimension=3)

    X3 = SingleQubitGateToQutritGate(cirq.X)
    CNOT3 = TwoQubitGateToQutritGate(cirq.CNOT)
    ideal_matrix = np.zeros((9, 9), dtype=np.complex128)
    ideal_matrix[4, 4] = 1.0  # |11>

    return qutrits, cirq.Circuit(X3(qutrits[0]), CNOT3(qutrits[0], qutrits[1])), ideal_matrix


def test_wrapped_execution(args):
    qutrits, circtrit, ideal_matrix = args

    sim = cirq.DensityMatrixSimulator()

    final_density_matrix = sim.simulate(circtrit).final_density_matrix
    assert np.allclose(final_density_matrix, ideal_matrix)


def test_qutrit_noise_model(args):
    qutrits, circtrit, ideal_matrix = args

    p_1 = 0.001 / 3
    p_2 = 0.01 / 15
    noise_model = GokhaleNoiseModelOnQutrits(
        single_qutrit_error_weights=[1 - p_1] + 8 * [p_1 / 8],
        two_qutrit_error_weights=[1 - p_2] + 80 * [p_2 / 80],
        lambda_short=100.0 / 10000.0,
        lambda_long=300.0 / 10000.0,
    )
    noisy_sim = cirq.DensityMatrixSimulator(noise=noise_model)
    noisy_result = noisy_sim.simulate(circtrit).final_density_matrix
    assert not np.allclose(noisy_result, ideal_matrix)
    assert np.isclose(np.trace(noisy_result), 1.0), np.trace(noisy_result)
    print(
        "Resulting fidelity:", cirq.qis.fidelity(ideal_matrix, noisy_result, (len(ideal_matrix),))
    )


def test_parameterized_noise(args):
    qutrits, circtrit, ideal_matrix = args
    edges = [(qutrits[0], qutrits[1]), (qutrits[1], qutrits[2])]
    single_qutrit_error_dict = dict()
    two_qutrit_error_dict = dict()
    p_1 = 0.001 / 3
    p_2 = 0.01 / 15
    for q in qutrits:
        single_qutrit_error_dict[q] = np.random.uniform(
            0, p_1
        )  # initialize single body gate errors
        two_qutrit_error_dict[q] = dict()  # initialize next level of dicts
    for e in edges:
        edge_error = np.random.uniform(0, p_2)
        two_qutrit_error_dict[e[0]][e[1]] = edge_error
        two_qutrit_error_dict[e[1]][
            e[0]
        ] = edge_error  # set two-body gate errors for both directions

    noise_model = HardwareAwareSymmetricNoise(
        single_qutrit_hardware_error_rates=single_qutrit_error_dict,
        two_qutrit_hardware_error_rates=two_qutrit_error_dict,
        lambda_short=100.0 / 10000.0,
        lambda_long=300.0 / 10000.0,
    )

    noisy_sim = cirq.DensityMatrixSimulator(noise=noise_model)
    noisy_result = noisy_sim.simulate(circtrit).final_density_matrix
    assert not np.allclose(noisy_result, ideal_matrix)
    assert np.isclose(np.trace(noisy_result), 1.0), np.trace(noisy_result)
    print(
        "Resulting fidelity:", cirq.qis.fidelity(ideal_matrix, noisy_result, (len(ideal_matrix),))
    )
