# Cirqtrit

Cirqtrit is an exploration into placement heuristics for the placement of qubits in a circuit compiled 
using qutrit-assisted decompositions, with simulation of qutrit-level noise for the purposes of benchmarking 
against equivalent compilation methods that do not use such decompositions.

## Installation
Cirqtrit relies on Cirq, Numpy, Pytket, Pytket-Cirq, and Pytest for its base imports.
These can be installed by running the following command in the same directory as `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Usage

To see a demonstration of hardware aware noise model in simulation, use the command:

```python
python noise_demo.py
```

To see a demonstration of routing for gates it is currently possible to route, use the command:
```python
python routing_demo.py
```


## License
[MIT](https://choosealicense.com/licenses/mit/)

## Open-Source Credit
With thanks to Cambridge Quantum Computing for their open-source `pytket-cirq` conversion code,
modified for extension and used under the same license,

and thanks to the Cirq Developers for providing the ternary Pauli operators and experimental data 
used for development of our noise model, as well as the codebase on which we extend.