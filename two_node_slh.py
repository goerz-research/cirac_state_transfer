"""Routines the construct the SLH model for a two-node network"""
import sympy

from single_channel_slh import qnet_node_system, node_hamiltonian

from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import Destroy, LocalSigma
from qnet.algebra.circuit_algebra import (
    SLH, connect, identity_matrix, CircuitSymbol)



def setup_qnet_sys(n_cavity, stark_shift=False, zero_phi=True,
        keep_delta=False, slh=True):
    """Return the effective SLH model (if `slh` is True) or a the symbolic
    circuit (if `slh` is False) for a two-node network, together with the
    symbols and operators in each node"""
    Sym1, Op1 = qnet_node_system('1', n_cavity,
                                 zero_phi=zero_phi, keep_delta=keep_delta)
    Sym2, Op2 = qnet_node_system('2', n_cavity,
                                 zero_phi=zero_phi, keep_delta=keep_delta)
    if slh:
        H1 = node_hamiltonian(Sym1, Op1,
                              stark_shift=stark_shift, zero_phi=zero_phi,
                              keep_delta=keep_delta)
        H2 = node_hamiltonian(Sym2, Op2,
                              stark_shift=stark_shift, zero_phi=zero_phi,
                              keep_delta=keep_delta)
        S = identity_matrix(1)
        κ = Sym1['kappa']
        L1 = sympy.sqrt(2*κ) * Op1['a']
        κ = Sym2['kappa']
        L2 = sympy.sqrt(2*κ) * Op2['a']
        SLH1 = SLH(S, [L1,], H1)
        SLH2 = SLH(S, [L2,], H2)

    Node1 = CircuitSymbol("Node1", cdim=1)
    Node2 = CircuitSymbol("Node2", cdim=1)

    components = [Node1, Node2]
    connections = [ ((Node1, 0), (Node2, 0)), ]
    circuit = connect(components, connections)
    if slh:
        network = circuit.substitute({Node1: SLH1, Node2: SLH2})
    else:
        network = circuit

    return network, Sym1, Op1, Sym2, Op2
