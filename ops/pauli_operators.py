# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

# All constants in this with credit to
# https://github.com/epiqc/qutrits/blob/4ae948006c4ff3d47ae47e69d05dcd3025485665/cirq/circuits/circuit.py#L1493
X3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.complex128)
Z3 = np.array(
    [
        [1, 0, 0],
        [0, np.e ** (np.pi * 1j * -2.0 / 3.0), 0],
        [0, 0, np.e ** (np.pi * 1j * -4.0 / 3.0)],
    ],
    dtype=np.complex128,
)
Y3 = X3 @ Z3
V3 = X3 @ Z3 @ Z3

single_qutrit_pauli_operators = [
    np.eye(3),
    Z3,
    Z3 @ Z3,
    X3,
    X3 @ Z3,
    X3 @ Z3 @ Z3,
    X3 @ X3,
    X3 @ X3 @ Z3,
    X3 @ X3 @ Z3 @ Z3,
]
two_qutrit_pauli_operators = [
    np.kron(np.eye(3), np.eye(3)),
    np.kron(np.eye(3), Z3),
    np.kron(np.eye(3), Z3 @ Z3),
    np.kron(np.eye(3), X3),
    np.kron(np.eye(3), X3 @ Z3),
    np.kron(np.eye(3), X3 @ Z3 @ Z3),
    np.kron(np.eye(3), X3 @ X3),
    np.kron(np.eye(3), X3 @ X3 @ Z3),
    np.kron(np.eye(3), X3 @ X3 @ Z3 @ Z3),
    np.kron(Z3, np.eye(3)),
    np.kron(Z3, Z3),
    np.kron(Z3, Z3 @ Z3),
    np.kron(Z3, X3),
    np.kron(Z3, X3 @ Z3),
    np.kron(Z3, X3 @ Z3 @ Z3),
    np.kron(Z3, X3 @ X3),
    np.kron(Z3, X3 @ X3 @ Z3),
    np.kron(Z3, X3 @ X3 @ Z3 @ Z3),
    np.kron(Z3 @ Z3, np.eye(3)),
    np.kron(Z3 @ Z3, Z3),
    np.kron(Z3 @ Z3, Z3 @ Z3),
    np.kron(Z3 @ Z3, X3),
    np.kron(Z3 @ Z3, X3 @ Z3),
    np.kron(Z3 @ Z3, X3 @ Z3 @ Z3),
    np.kron(Z3 @ Z3, X3 @ X3),
    np.kron(Z3 @ Z3, X3 @ X3 @ Z3),
    np.kron(Z3 @ Z3, X3 @ X3 @ Z3 @ Z3),
    np.kron(X3, np.eye(3)),
    np.kron(X3, Z3),
    np.kron(X3, Z3 @ Z3),
    np.kron(X3, X3),
    np.kron(X3, X3 @ Z3),
    np.kron(X3, X3 @ Z3 @ Z3),
    np.kron(X3, X3 @ X3),
    np.kron(X3, X3 @ X3 @ Z3),
    np.kron(X3, X3 @ X3 @ Z3 @ Z3),
    np.kron(X3 @ Z3, np.eye(3)),
    np.kron(X3 @ Z3, Z3),
    np.kron(X3 @ Z3, Z3 @ Z3),
    np.kron(X3 @ Z3, X3),
    np.kron(X3 @ Z3, X3 @ Z3),
    np.kron(X3 @ Z3, X3 @ Z3 @ Z3),
    np.kron(X3 @ Z3, X3 @ X3),
    np.kron(X3 @ Z3, X3 @ X3 @ Z3),
    np.kron(X3 @ Z3, X3 @ X3 @ Z3 @ Z3),
    np.kron(X3 @ Z3 @ Z3, np.eye(3)),
    np.kron(X3 @ Z3 @ Z3, Z3),
    np.kron(X3 @ Z3 @ Z3, Z3 @ Z3),
    np.kron(X3 @ Z3 @ Z3, X3),
    np.kron(X3 @ Z3 @ Z3, X3 @ Z3),
    np.kron(X3 @ Z3 @ Z3, X3 @ Z3 @ Z3),
    np.kron(X3 @ Z3 @ Z3, X3 @ X3),
    np.kron(X3 @ Z3 @ Z3, X3 @ X3 @ Z3),
    np.kron(X3 @ Z3 @ Z3, X3 @ X3 @ Z3 @ Z3),
    np.kron(X3 @ X3, np.eye(3)),
    np.kron(X3 @ X3, Z3),
    np.kron(X3 @ X3, Z3 @ Z3),
    np.kron(X3 @ X3, X3),
    np.kron(X3 @ X3, X3 @ Z3),
    np.kron(X3 @ X3, X3 @ Z3 @ Z3),
    np.kron(X3 @ X3, X3 @ X3),
    np.kron(X3 @ X3, X3 @ X3 @ Z3),
    np.kron(X3 @ X3, X3 @ X3 @ Z3 @ Z3),
    np.kron(X3 @ X3 @ Z3, np.eye(3)),
    np.kron(X3 @ X3 @ Z3, Z3),
    np.kron(X3 @ X3 @ Z3, Z3 @ Z3),
    np.kron(X3 @ X3 @ Z3, X3),
    np.kron(X3 @ X3 @ Z3, X3 @ Z3),
    np.kron(X3 @ X3 @ Z3, X3 @ Z3 @ Z3),
    np.kron(X3 @ X3 @ Z3, X3 @ X3),
    np.kron(X3 @ X3 @ Z3, X3 @ X3 @ Z3),
    np.kron(X3 @ X3 @ Z3, X3 @ X3 @ Z3 @ Z3),
    np.kron(X3 @ X3 @ Z3 @ Z3, np.eye(3)),
    np.kron(X3 @ X3 @ Z3 @ Z3, Z3),
    np.kron(X3 @ X3 @ Z3 @ Z3, Z3 @ Z3),
    np.kron(X3 @ X3 @ Z3 @ Z3, X3),
    np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ Z3),
    np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ Z3 @ Z3),
    np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ X3),
    np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ X3 @ Z3),
    np.kron(X3 @ X3 @ Z3 @ Z3, X3 @ X3 @ Z3 @ Z3),
]
