# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [ Modified May 2nd, 2022 by Ray Zhang ]


from typing import Dict
import cirq.ops
import cirq_google
from pytket.circuit import OpType

# For translating cirq circuits to tket circuits
cirq_common = cirq.ops.common_gates
cirq_pauli = cirq.ops.pauli_gates

cirq_CH = cirq_common.H.controlled(1)

# map cirq common gates to pytket gates
_cirq2ops_mapping = {
    cirq_common.CNOT: OpType.CX,
    cirq_common.H: OpType.H,
    cirq_common.MeasurementGate: OpType.Measure,
    cirq_common.XPowGate: OpType.Rx,
    cirq_common.YPowGate: OpType.Ry,
    cirq_common.ZPowGate: OpType.Rz,
    cirq_common.XPowGate(exponent=0.5): OpType.V,
    cirq_common.XPowGate(exponent=-0.5): OpType.Vdg,
    cirq_common.S: OpType.S,
    cirq_common.SWAP: OpType.SWAP,
    cirq_common.T: OpType.T,
    cirq_pauli.X: OpType.X,
    cirq_pauli.Y: OpType.Y,
    cirq_pauli.Z: OpType.Z,
    cirq.ops.I: OpType.noop,
    cirq_common.CZPowGate: OpType.CU1,
    cirq_common.CZ: OpType.CZ,
    cirq_CH: OpType.CH,
    cirq.ops.CSwapGate: OpType.CSWAP,
    cirq_common.ISwapPowGate: OpType.ISWAP,
    cirq_common.ISWAP: OpType.ISWAPMax,
    cirq.ops.FSimGate: OpType.FSim,
    cirq_google.SYC: OpType.Sycamore,
    cirq.ops.parity_gates.ZZPowGate: OpType.ZZPhase,
    cirq.ops.parity_gates.XXPowGate: OpType.XXPhase,
    cirq.ops.parity_gates.YYPowGate: OpType.YYPhase,
    cirq.ops.PhasedXPowGate: OpType.PhasedX,
    cirq.ops.PhasedISwapPowGate: OpType.PhasedISWAP,
}
# reverse mapping for convenience
_ops2cirq_mapping: Dict = dict((item[1], item[0]) for item in _cirq2ops_mapping.items())
# spot special rotation gates
_constant_gates = (
    cirq_common.CNOT,
    cirq_common.H,
    cirq_common.S,
    cirq_common.SWAP,
    cirq_common.T,
    cirq_pauli.X,
    cirq_pauli.Y,
    cirq_pauli.Z,
    cirq_common.CZ,
    cirq_CH,
    cirq_common.ISWAP,
    cirq_google.SYC,
    cirq.ops.I,
)
_rotation_types = (
    cirq_common.XPowGate,
    cirq_common.YPowGate,
    cirq_common.ZPowGate,
    cirq_common.CZPowGate,
    cirq_common.ISwapPowGate,
    cirq.ops.parity_gates.ZZPowGate,
    cirq.ops.parity_gates.XXPowGate,
    cirq.ops.parity_gates.YYPowGate,
)
