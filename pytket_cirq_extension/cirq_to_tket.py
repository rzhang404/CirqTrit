# Copyright 2019-2022 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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


from typing import List, Dict, FrozenSet, cast, Any, Union
import cmath
from cirq.devices import LineQubit, GridQubit
import cirq.ops
from pytket.circuit import Circuit, OpType, Qubit, Bit
from sympy import pi, Basic, Symbol  # type: ignore
from pytket_cirq_extension.conversion_mappings import (
    cirq_common,
    cirq_pauli,
    _constant_gates,
    _cirq2ops_mapping,
)
from pytket.circuit import CustomGateDef, CustomGate
from ops.to_qubit_wrappers import SingleQutritGateToQubitGate, TwoQutritGateToQubitGate


def cirq_to_tk(circuit: cirq.circuits.Circuit) -> Circuit:
    """Converts a Cirq :py:class:`Circuit` to a tket :py:class:`Circuit` object.

    :param circuit: The input Cirq :py:class:`Circuit`

    :raises NotImplementedError: If the input contains a Cirq :py:class:`Circuit`
        operation which is not yet supported by pytket

    :return: The tket :py:class:`Circuit` corresponding to the input circuit
    """
    tkcirc = Circuit()
    qmap = {}
    for qb in circuit.all_qubits():
        if isinstance(qb, LineQubit):
            uid = Qubit("q", qb.x)
        elif isinstance(qb, GridQubit):
            uid = Qubit("g", qb.row, qb.col)
        elif isinstance(qb, cirq.ops.NamedQubit):
            uid = Qubit(qb.name)
        else:
            raise NotImplementedError("Cannot convert qubits of type " + str(type(qb)))
        tkcirc.add_qubit(uid)
        qmap.update({qb: uid})
    for moment in circuit:
        for op in moment.operations:
            gate = op.gate
            gatetype = type(gate)
            qb_lst = [qmap[q] for q in op.qubits]

            if isinstance(gate, cirq.ops.global_phase_op.GlobalPhaseGate):
                tkcirc.add_phase(cmath.phase(gate.coefficient) / pi)
                continue
            if isinstance(gate, cirq_common.HPowGate) and gate.exponent == 1:
                gate = cirq_common.H
            elif (
                gatetype == cirq_common.CNotPowGate
                and cast(cirq_common.CNotPowGate, gate).exponent == 1
            ):
                gate = cirq_common.CNOT
            elif gatetype == cirq_pauli._PauliX and cast(cirq_pauli._PauliX, gate).exponent == 1:
                gate = cirq_pauli.X
            elif gatetype == cirq_pauli._PauliY and cast(cirq_pauli._PauliY, gate).exponent == 1:
                gate = cirq_pauli.Y
            elif gatetype == cirq_pauli._PauliZ and cast(cirq_pauli._PauliZ, gate).exponent == 1:
                gate = cirq_pauli.Z

            apply_in_parallel = False
            if isinstance(gate, cirq.ops.ParallelGate):
                if gate.num_copies != len(qb_lst):
                    raise NotImplementedError("ParallelGate parameters defined incorrectly.")
                gate = gate.sub_gate
                gatetype = type(gate)
                apply_in_parallel = True

            if gate in _constant_gates:
                try:
                    optype = _cirq2ops_mapping[gate]
                except KeyError as error:
                    raise NotImplementedError(
                        "Operation not supported by tket: " + str(op.gate)
                    ) from error
                params: List[Union[float, Basic, Symbol]] = []
            elif isinstance(gate, cirq_common.MeasurementGate):
                uid = Bit(gate.key)
                tkcirc.add_bit(uid)
                tkcirc.Measure(*qb_lst, uid)
                continue
            elif isinstance(gate, cirq.ops.PhasedXPowGate):
                optype = OpType.PhasedX
                pe = gate.phase_exponent
                params = [gate.exponent, pe]
            elif isinstance(gate, cirq.ops.FSimGate):
                optype = OpType.FSim
                params = [gate.theta / pi, gate.phi / pi]
            elif isinstance(gate, cirq.ops.PhasedISwapPowGate):
                optype = OpType.PhasedISWAP
                params = [gate.phase_exponent, gate.exponent]
            elif isinstance(gate, SingleQutritGateToQubitGate):
                dummy_sub_circ = Circuit()
                gate_def = CustomGateDef.define(str(gate.base_gate), dummy_sub_circ, [])
                tkcirc.add_custom_gate(gate_def, [], qb_lst)
                continue
            elif isinstance(gate, TwoQutritGateToQubitGate):
                dummy_sub_circ = Circuit()
                gate_def = CustomGateDef.define(str(gate.base_gate), dummy_sub_circ, [])
                tkcirc.add_custom_gate(gate_def, [], qb_lst)
                continue
            else:
                try:
                    optype = _cirq2ops_mapping[gatetype]
                    params = [cast(Any, gate).exponent]
                except (KeyError, AttributeError) as error:
                    raise NotImplementedError(
                        "Operation not supported by tket: " + str(op.gate)
                    ) from error
            if apply_in_parallel:
                for qb in qb_lst:
                    tkcirc.add_gate(optype, params, [qb])
            else:
                tkcirc.add_gate(optype, params, qb_lst)
    return tkcirc
