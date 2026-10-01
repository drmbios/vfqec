"""Qiskit coherent layers and one-round, 17-qubit surface syndrome circuits."""

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from vfqec.physics.operators import initial_state


def append_gates(circuit: QuantumCircuit, gates) -> None:
    for gate in gates:
        if gate.twirl:
            raise ValueError("A twirl is an ensemble, not one deterministic unitary")
        if len(gate.support) == 2:
            circuit.rzz(gate.angle, *gate.support)
        else:
            getattr(circuit, "r" + gate.axis.lower())(gate.angle, gate.support[0])


def coherent_circuit(n: int, gates) -> QuantumCircuit:
    circuit = QuantumCircuit(n)
    append_gates(circuit, gates)
    return circuit


def syndrome_circuit(code, gates=(), measure: bool = True) -> QuantumCircuit:
    """Ideal state preparation/extraction; phenomenological p/q are applied by the plant."""
    circuit = QuantumCircuit(code.n + code.m, code.m if measure else 0)
    circuit.append(StatePreparation(initial_state(code)), range(code.n))
    append_gates(circuit, gates)
    for j, (axis, support) in enumerate(code.checks):
        anc = code.n + j
        if axis == "X":
            circuit.h(anc)
            for data in support:
                circuit.cx(anc, data)
            circuit.h(anc)
        else:
            for data in support:
                circuit.cx(data, anc)
        if measure:
            circuit.measure(anc, j)
    return circuit
