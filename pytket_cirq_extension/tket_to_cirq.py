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


import cmath
from logging import warning
from cirq.devices import LineQubit, GridQubit
import cirq.ops
from pytket.circuit import Circuit, OpType
from sympy import pi
from pytket_cirq_extension.conversion_mappings import _ops2cirq_mapping


def tk_to_cirq(tkcirc: Circuit, copy_all_qubits: bool = False) -> cirq.circuits.Circuit:
    """Converts a tket :py:class:`Circuit` object to a Cirq :py:class:`Circuit`.

    :param tkcirc: The input tket :py:class:`Circuit`

    :return: The Cirq :py:class:`Circuit` corresponding to the input circuit
    """
    if copy_all_qubits:
        tkcirc = tkcirc.copy()
        for q in tkcirc.qubits:
            tkcirc.add_gate(OpType.noop, [q])

    qmap = {}
    line_name = None
    grid_name = None
    # Since Cirq can only support registers of up to 2 dimensions, we explicitly
    # check for 3-dimensional registers whose third dimension is trivial.
    # SquareGrid architectures are of this form.
    indices = [qb.index for qb in tkcirc.qubits]
    is_flat_3d = all(idx[2] == 0 for idx in indices if len(idx) == 3)
    for qb in tkcirc.qubits:
        if len(qb.index) == 0:
            qmap.update({qb: cirq.ops.NamedQubit(qb.reg_name)})
        elif len(qb.index) == 1:
            if line_name != None and line_name != qb.reg_name:
                raise NotImplementedError("Cirq can only support a single linear register")
            line_name = qb.reg_name
            qmap.update({qb: LineQubit(qb.index[0])})
        elif len(qb.index) == 2 or (len(qb.index) == 3 and is_flat_3d):
            if grid_name != None and grid_name != qb.reg_name:
                raise NotImplementedError("Cirq can only support a single grid register")
            grid_name = qb.reg_name
            qmap.update({qb: GridQubit(qb.index[0], qb.index[1])})
        else:
            raise NotImplementedError("Cirq can only support registers of dimension <=2")
    oplst = []
    for command in tkcirc:
        op = command.op
        optype = op.type
        try:
            gatetype = _ops2cirq_mapping[optype]
        except KeyError as error:
            raise NotImplementedError(
                "Cannot convert tket Op to Cirq gate: " + op.get_name()
            ) from error
        if optype == OpType.Measure:
            qid = qmap[command.args[0]]
            bit = command.args[1]
            cirqop = cirq.ops.measure(qid, key=bit.__repr__())
        else:
            qids = [qmap[qbit] for qbit in command.args]
            params = op.params
            if len(params) == 0:
                cirqop = gatetype(*qids)
            elif optype == OpType.PhasedX:
                cirqop = gatetype(phase_exponent=params[1], exponent=params[0])(*qids)
            elif optype == OpType.FSim:
                cirqop = gatetype(theta=float(params[0] * pi), phi=float(params[1] * pi))(*qids)
            elif optype == OpType.PhasedISWAP:
                cirqop = gatetype(phase_exponent=params[0], exponent=params[1])(*qids)
            else:
                cirqop = gatetype(exponent=params[0])(*qids)
        oplst.append(cirqop)
    try:

        coeff = cmath.exp(float(tkcirc.phase) * cmath.pi * 1j)
        if coeff.real < 1e-8:  # tolerance permitted by cirq for GlobalPhaseGate
            coeff = coeff.imag * 1j
        if coeff.imag < 1e-8:
            coeff = coeff.real
        if coeff != 1.0:
            oplst.append(cirq.global_phase_operation(coeff))
    except ValueError:
        warning("Global phase is dependent on a symbolic parameter, so cannot adjust for " "phase")
    return cirq.circuits.Circuit(*oplst)
