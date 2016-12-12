"""Routines the construct the SLH model for a two-node network"""
import sympy

from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import Destroy, LocalSigma
from qnet.algebra.circuit_algebra import (
    SLH, connect, identity_matrix, CircuitSymbol)


def dagger(op):
    return op.adjoint()


def node_hamiltonian(Sym, Op, stark_shift=False, zero_phi=True,
    keep_delta=False):
    """Hamiltonian for a single node, in the RWA

    The "simplified" form that can be used for numerics is for the default
    parameters. The expanded form that corresponds to Eq. 2 in the paper is for
    stark_shift=True, zero_phi=False, keep_delta=True
    """
    # Symbols
    Δ, g, Ω, I = (Sym['Delta'], Sym['g'], Sym['Omega'], Sym['I'])
    if not zero_phi:
        exp = Sym['exp']
        φ = Sym['phi']
    if keep_delta:
        δ = Sym['delta']
    else:
        δ = g**2 / Δ
    # Cavity operators
    Op_a = Op['a']; Op_a_dag = dagger(Op_a); Op_n = Op_a_dag * Op_a
    # Qubit operators
    Op_gg = Op['|g><g|']; Op_ee = Op['|e><e|']; Op_eg = Op['|e><g|']
    Op_ge = dagger(Op_eg)
    # Hamiltonian
    if zero_phi:
        H = -δ*Op_n + (g**2/Δ)*Op_n*Op_gg \
            -I * (g/(2*Δ)) * Ω * (Op_eg*Op_a - Op_ge*Op_a_dag)
    else:
        H = -δ*Op_n + (g**2/Δ)*Op_n*Op_gg \
            -I * (g/(2*Δ)) * Ω * (
                exp(I*φ) * Op_eg*Op_a - exp(-I*φ) * Op_ge*Op_a_dag)
    if stark_shift:
        H += ((Ω**2)/(4*Δ))*Op_ee
    return H


def qnet_node_system(node_index, n_cavity, zero_phi=True, keep_delta=False):
    """Define symbols and operators for a single node"""
    from sympy import symbols
    HilAtom = LocalSpace('q%d' % int(node_index), basis=('g', 'e'))
    HilCavity = LocalSpace('c%d' % int(node_index), dimension=n_cavity)
    Sym = {}
    Sym['Delta'] = symbols(r'Delta_%s' % node_index, real=True)
    Sym['g'] = symbols(r'g_%s' % node_index, positive=True)
    Sym['Omega'] = symbols(r'Omega_%s' % node_index)
    Sym['I'] = sympy.I
    Sym['kappa'] = sympy.symbols(r'kappa', positive=True)
    if not zero_phi:
        Sym['phi'] = sympy.symbols(r'phi_%s' % node_index, real=True)
        Sym['exp'] = sympy.exp
    if keep_delta:
        Sym['delta'] = symbols(r'delta_%s' % node_index, real=True)
    Op = {}
    Op['a'] = Destroy(hs=HilCavity)
    Op['|g><g|'] = LocalSigma('g', 'g', hs=HilAtom)
    Op['|e><e|'] = LocalSigma('e', 'e', hs=HilAtom)
    Op['|e><g|'] = LocalSigma('e', 'g', hs=HilAtom)
    return Sym, Op


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
