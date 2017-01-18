"""Routines the construct the SLH model for a network of JC cavities with two
I/O channels"""

import sympy
from sympy import symbols
from single_channel_slh import qnet_node_system, node_hamiltonian
from qnet.algebra.circuit_algebra import (
    SLH, connect, identity_matrix, CircuitSymbol)

def slh_map_2chan_chain(components, n_cavity):
    n_nodes = len(components)
    slh_map = {}
    for i in range(n_nodes):
        ind = i + 1
        Sym, Op = qnet_node_system(node_index='%d' % ind,
                                    n_cavity=n_cavity)
        S = identity_matrix(2)
        kappa_l, kappa_r = symbols("kappa_%dl, kappa_%dr" % (ind, ind),
                                    positive=True)
        L = [sympy.sqrt(2*kappa_l) * Op['a'],
                sympy.sqrt(2*kappa_r) * Op['a']]
        H = node_hamiltonian(Sym, Op)
        slh_map[components[i]] = SLH(S, L, H)
    return slh_map


def setup_qnet_2chan_chain(n_cavity, n_nodes, slh=True, topology=None):
    """Set up a chain of JC system with two channels

    Args:
        n_cavity (int): Number of levels in the cavity (numerical truncation)
        n_nodes (int):  Number of nodes in the chain
        slh (bool): If True, return effective SLH object. If False, return
            circuit of linked nodes
        topology (str or None): How the nodes should be linked up, see below

    Notes:

        The `topology` can take the following values:
        * None (default): chain is open-endeded. The total system will have two
          I/O channels
        * "FB": The rightmost output is fed back into the rightmost input::
              >-------+
                      |
              <-------+
    """

    # Set up the circuit
    components = []
    connections = []
    prev_node = None
    for i in range(n_nodes):
        ind = i + 1
        cur_node = CircuitSymbol("Node%d" % ind, cdim=2)
        components.append(cur_node)
        if prev_node is not None:
            if topology in [None, 'FB']:
                connections.append(((prev_node, 1), (cur_node, 0)))
                connections.append(((cur_node, 0), (prev_node, 1)))
            else:
                raise ValueError("Unknown topology: %s" % topology)
        prev_node = cur_node
    if topology == 'FB':
        connections.append(((cur_node, 1), (cur_node, 1)))
    circuit = connect(components, connections)

    if slh:
        slh_map = slh_map_2chan_chain(components, n_cavity)
        return circuit.substitute(slh_map).toSLH()
    else:
        return circuit

