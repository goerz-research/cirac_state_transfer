"""Test the model"""

from os.path import join
from collections import OrderedDict

import numpy as np
import pytest
import qnet
import QDYN

# builtin fixture: tmpdir


@pytest.fixture
def two_node_slh():
    """SLH, Symbols, and Operators for two-node network"""
    from two_node_slh import setup_qnet_sys, qnet_node_system
    slh, Sym1, Op1, Sym2, Op2 = setup_qnet_sys(n_cavity=3)
    return slh, Sym1, Op1, Sym2, Op2


def test_setup_qnet_sys(two_node_slh):
    """Test that setup_qnet_sys returns an SLH object in the correct Hilbert
    space"""
    slh, Sym1, Op1, Sym2, Op2 = two_node_slh
    assert isinstance(slh, qnet.algebra.circuit_algebra.SLH)
    assert len(slh.space.local_factors()) == 4


def test_make_qdyn_model(two_node_slh, tmpdir):
    """Test that we can build a LevelModel for the two-node network"""
    from two_node_qdyn import make_qdyn_model, state
    sys, Sym1, Op1, Sym2, Op2 = two_node_slh
    pulse1 = QDYN.pulse.Pulse(QDYN.pulse.pulse_tgrid(4, 1000),
                              time_unit='microsec', ampl_unit='MHz')
    pulse2 = QDYN.pulse.Pulse(QDYN.pulse.pulse_tgrid(4, 1000),
                              time_unit='microsec', ampl_unit='MHz')
    Delta = 5000
    g = 50
    kappa = 0.5

    states = OrderedDict([
        ('01', state(sys, 0, 1, 0, 0, fmt='qutip')),
        ('10', state(sys, 1, 0, 0, 0, fmt='numpy'))
    ])

    model = make_qdyn_model(sys, Delta, g, kappa, Sym1, Op1, Sym2, Op2, pulse1,
                            pulse2, states=states)
    rf = str(tmpdir.join("rf"))
    model.write_to_runfolder(rf)

    H0 = QDYN.io.read_indexed_matrix(join(rf, 'H0.dat'),
                                     expand_hermitian=False)
    assert H0.shape[0] == H0.shape[1] == 36

    psi01 = QDYN.state.read_psi_amplitudes(join(rf, 'psi_01.dat'), n=36)
    assert np.max(np.abs(psi01 - QDYN.linalg.vectorize(states['01']))) < 1e-14
    psi10 = QDYN.state.read_psi_amplitudes(join(rf, 'psi_10.dat'), n=36)
    assert np.max(np.abs(psi10 - QDYN.linalg.vectorize(states['10']))) < 1e-14
