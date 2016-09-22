"""Checks on OCT runfolders"""

from os.path import join

import numpy as np
import QDYN

from two_node_qdyn import state


def check_two_oct_pulses(config):
    """Check that config data contains two pulses with identical OCT parameters
    and a flattop shape"""
    pulse1_config = config['pulse'][0]
    pulse2_config = config['pulse'][1]
    for key, val in pulse1_config.items():
        if key not in ['id', 'oct_outfile', 'filename']:
            assert pulse2_config[key] == val
    delta = config['tgrid']['t_start'] - config['pulse'][0]['shape_t_start']
    assert abs(float(delta)) < 1e-14
    delta = config['tgrid']['t_stop'] - config['pulse'][0]['shape_t_stop']
    assert abs(float(delta)) < 1e-14
    assert config['pulse'][0]['oct_shape'] == 'flattop'


def check_01_10_state(rf, sys, n_cavity):
    """Check that the config file defines exactly two states, 10, and 01"""
    config = QDYN.config.read_config_file(join(rf, 'config'))
    n_hilbert = 2 * 2 * n_cavity * n_cavity

    psi01 = state(sys, 0, 1, 0, 0, fmt='numpy')
    psi10 = state(sys, 1, 0, 0, 0, fmt='numpy')

    assert config['psi'][0]['filename'] == 'psi_10.dat'
    file_psi1 = join(rf, config['psi'][0]['filename'])
    ampl_psi1 = QDYN.state.read_psi_amplitudes(file_psi1, n=n_hilbert)
    assert np.max(np.abs(ampl_psi1 - psi10)) < 1e-14

    assert config['psi'][1]['filename'] == 'psi_01.dat'
    file_psi2 = join(rf, config['psi'][1]['filename'])
    ampl_psi2 = QDYN.state.read_psi_amplitudes(file_psi2, n=n_hilbert)
    assert np.max(np.abs(ampl_psi2 - psi01)) < 1e-14


def check_oct_bidir(rf, sys, n_cavity):
    """Check consistency of OCT folder for bidirectional control problem"""
    config = QDYN.config.read_config_file(join(rf, 'config'))
    assert config['user_strings']['initial_states'] == '01,10'
    assert config['user_strings']['target_states'] == '10,01'
    check_two_oct_pulses(config)
    check_01_10_state(rf, sys, n_cavity)


def check_oct_bidir_liouville(rf, sys, n_cavity):
    """check bidirectional control problem for direct optimization in
    Liouville space"""
    check_oct_bidir(rf, sys, n_cavity)
    config = QDYN.config.read_config_file(join(rf, 'config'))
    assert 'observables' not in config
    assert not config['prop']['use_mcwf']

def check_oct_bidir_mcwf(rf, sys, n_cavity):
    """check bidirectional control problem for optimization using MCWF
    trajectories"""
    check_oct_bidir(rf, sys, n_cavity)
    config = QDYN.config.read_config_file(join(rf, 'config'))
    assert 'observables' not in config
    assert config['prop']['use_mcwf']


def check_oct_unidir(rf, sys, n_cavity):
    """Check consistency of OCT folder for unidirectional control problem"""
    config = QDYN.config.read_config_file(join(rf, 'config'))
    assert config['user_strings']['initial_states'] == '10'
    assert config['user_strings']['target_states'] == '01'
    check_two_oct_pulses(config)
    check_01_10_state(rf, sys, n_cavity)


def check_oct_unidir_liouville(rf, sys, n_cavity):
    """check unidirectional control problem for direct optimization in
    Liouville space"""
    check_oct_unidir(rf, sys, n_cavity)
    config = QDYN.config.read_config_file(join(rf, 'config'))
    assert 'observables' not in config
    assert not config['prop']['use_mcwf']


def check_oct_unidir_mcwf(rf, sys, n_cavity):
    """check bidirectional control problem for optimization using MCWF
    trajectories"""
    check_oct_unidir(rf, sys, n_cavity)
    config = QDYN.config.read_config_file(join(rf, 'config'))
    assert 'observables' not in config
    assert config['prop']['use_mcwf']
