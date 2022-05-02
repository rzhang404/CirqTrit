import tempfile
import json
from pytket import Circuit, OpType
from pytket.circuit import CustomGateDef, CustomGate

def_circ = Circuit(2)
gate_def = CustomGateDef.define("C+", def_circ, [])
circ = Circuit(3)
circ.add_custom_gate(gate_def, [], [0, 1])
circ.add_custom_gate(gate_def, [], [0, 2])

circ_dict = circ.to_dict()
print(circ_dict)
print("\n")

with tempfile.TemporaryFile("w+") as fp:
    json.dump(circ_dict, fp)
    fp.seek(0)
    new_circ = Circuit.from_dict(json.load(fp))

comms = new_circ.get_commands()

print(comms)
