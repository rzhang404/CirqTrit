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

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

[comment]: <> (todo: credit epiqc for kraus operators)

### Noise Models


## License
[MIT](https://choosealicense.com/licenses/mit/)

## Open-Source Credit
With thanks to Cambridge Quantum Computing for their open-source `pytket-cirq` conversion code,
modified for extension and used under the same license.