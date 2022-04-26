import cirq
import numpy as np
from prototype.qutrit_gates import SingleQubitGateToQutritGate, TwoQubitGateToQutritGate
from prototype.noise_model import GokhaleNoiseModelOnQutrits, HardwareAwareSymmetricNoise
import pytest

@pytest.fixture
def args():
    qutrits = cirq.LineQid.range(3, dimension=3)

    X3 = SingleQubitGateToQutritGate(cirq.X)
    CNOT3 = TwoQubitGateToQutritGate(cirq.CNOT)
    ideal_matrix = np.zeros((9, 9), dtype=np.complex128)
    ideal_matrix[4, 4] = 1.0  # |11>

    return qutrits, cirq.Circuit(
        X3(qutrits[0]),
        CNOT3(qutrits[0], qutrits[1])
    ), ideal_matrix


def test_wrapped_execution(args):
    qutrits, circtrit, ideal_matrix = args

    sim = cirq.DensityMatrixSimulator()

    final_density_matrix = sim.simulate(circtrit).final_density_matrix
    assert np.allclose(final_density_matrix, ideal_matrix)

def test_qutrit_noise_model(args):
    qutrits, circtrit, ideal_matrix = args

    p_1 = .001 / 3
    p_2 = .01 / 15
    # p_1 = 0
    # p_2 = 0
    noise_model = GokhaleNoiseModelOnQutrits(
        # gamma_1_short=1 - np.exp(-1 * 100.0 / 100000.0),
        # gamma_1_long=1 - np.exp(-1 * 300.0 / 100000.0),
        # gamma_2_short=1 - np.exp(-2 * 100.0 / 100000.0),
        # gamma_2_long=1 - np.exp(-2 * 300.0 / 100000.0),
        gamma_1_short=0,
        gamma_1_long=0,
        gamma_2_short=0,
        gamma_2_long=0,
        single_qutrit_error_weights=[1 - 8*p_1] + 8 * [p_1],
        two_qutrit_error_weights=[1 - 80*p_2] + 80 * [p_2],
    )
    noisy_sim = cirq.DensityMatrixSimulator(noise=noise_model)
    noisy_result = noisy_sim.simulate(circtrit).final_density_matrix
    assert not np.allclose(noisy_result, ideal_matrix)
    print(noisy_result, np.trace(noisy_result))
    assert np.isclose(np.trace(noisy_result), 1.0)
    print(cirq.qis.fidelity(ideal_matrix,
                            noisy_result,
                            (len(ideal_matrix),)))

def test_parameterized_noise(args):
    qutrits, circtrit, ideal_matrix = args
    edges = [(qutrits[0], qutrits[1]), (qutrits[1], qutrits[2])]
    single_qutrit_error_dict = dict()
    two_qutrit_error_dict = dict()
    p_1 = .001 / 3
    p_2 = .01 / 15
    for q in qutrits:
        single_qutrit_error_dict[q] = np.random.uniform(0, p_1)  # initialize single body gate errors
        # single_qutrit_error_dict[q] = np.random.random()
        two_qutrit_error_dict[q] = dict()  # initialize next level of dicts
    for e in edges:
        edge_error = np.random.uniform(0, p_2)
        # edge_error = np.random.random()
        two_qutrit_error_dict[e[0]][e[1]] = edge_error
        two_qutrit_error_dict[e[1]][e[0]] = edge_error  # set two-body gate errors for both directions

    noise_model = HardwareAwareSymmetricNoise(
        single_qutrit_error_rates=single_qutrit_error_dict,
        two_qutrit_error_rates=two_qutrit_error_dict
    )

    noisy_sim = cirq.DensityMatrixSimulator(noise=noise_model)
    noisy_result = noisy_sim.simulate(circtrit).final_density_matrix
    print(noisy_result)
    assert not np.allclose(noisy_result, ideal_matrix)
    print(noisy_result, np.trace(noisy_result))
    assert np.isclose(np.trace(noisy_result), 1.0)
