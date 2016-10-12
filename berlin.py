#!/usr/bin/env python
"""Generate data and plots for Berlin talk"""
from glob import glob
import sys
import os
from os.path import join
import shutil
import copy
from collections import OrderedDict
import QDYN
from QDYN.units import UnitFloat
import QDYN.shutil
from QDYN.shutil import mkdir
from two_node_slh import qnet_node_system, setup_qnet_sys
from two_node_qdyn import make_qdyn_model, state, err_state_to_state, prepare_rf_prop
from shwrapper import qdyn_optimize, qdyn_prop_traj, env

n_cavity = 2
SYS, Sym1, Op1, Sym2, Op2 = setup_qnet_sys(n_cavity=n_cavity)

Delta = 5000  # MHz
g     =   50  # MHz
kappa =  0.5  # MHz

psi01 = state(SYS, 0, 1, 0, 0)
psi10 = state(SYS, 1, 0, 0, 0)


def generate_analytical(rf='berlin_run/analytical'):
    if os.path.isfile(join(rf, 'qubit_pop.dat')):
        return
    mkdir(rf)
    omega1 = QDYN.pulse.Pulse.read("omega1_analytical.dat")
    omega2 = QDYN.pulse.Pulse.read("omega2_analytical.dat")
    analytical_model = make_qdyn_model(
        SYS, Delta, g, kappa, Sym1, Op1, Sym2, Op2,
        pulse1=omega1, pulse2=omega2, mcwf=True, non_herm=False,
        states={'': psi10}, set_observables=True)
    analytical_model.write_to_runfolder(rf)
    __ = qdyn_prop_traj(['--n-trajs=1', rf], _out=join(rf, 'prop.log'))


def oct_unidir_model(mcwf=True, non_herm=False, set_observables=True,
        lambda_a=1e-5, J_T_conv=1e-4, variation=None):
    from QDYN.pulse import blackman
    guess_omega1 = QDYN.pulse.Pulse.read("omega1_analytical.dat")
    guess_omega2 = QDYN.pulse.Pulse.read("omega2_analytical.dat")
    B = blackman(guess_omega1.tgrid, float(guess_omega1.t0),
                 float(guess_omega1.T))
    guess_omega1.amplitude = 70 * B
    guess_omega2.amplitude = 70* B
    if variation == 'short':
        guess_omega1 = QDYN.pulse.Pulse.read("omega1_short.dat") # compressed!
        guess_omega2 = QDYN.pulse.Pulse.read("omega2_short.dat")
        shape = QDYN.pulse.flattop(guess_omega1.tgrid, t_start=float(guess_omega1.t0),
                                   t_stop=float(guess_omega1.T),
                                   t_rise=float(0.1*guess_omega1.T))
        guess_omega1.amplitude *= 4*shape  # matches compression factor 4 in pulse
        guess_omega2.amplitude *= 4*shape
    states=OrderedDict([('10', psi10), ('01', psi01)])
    model = make_qdyn_model(
        SYS, Delta, g, kappa, Sym1, Op1, Sym2, Op2,
        pulse1=guess_omega1, pulse2=guess_omega2,
        mcwf=mcwf, non_herm=non_herm, states=states,
        set_observables=set_observables)
    pulse_settings = {
        guess_omega1: {
            'oct_shape': 'flattop',
            'shape_t_start': guess_omega1.t0, 'shape_t_stop': guess_omega1.T,
            't_rise': 0.1*guess_omega1.T, 't_fall': 0.1*guess_omega1.T,
            'oct_lambda_a' : lambda_a, 'oct_increase_factor': 5,
            'oct_outfile' : 'pulse1.oct.dat',
            'oct_pulse_max': UnitFloat(420, 'MHz'),
            'oct_pulse_min':  UnitFloat(-100, 'MHz')
        }
    }
    pulse_settings[guess_omega2] = copy.copy(pulse_settings[guess_omega1])
    pulse_settings[guess_omega2]['oct_outfile'] = 'pulse2.oct.dat'
    model.set_oct(pulse_settings, method='krotovpk', J_T_conv=J_T_conv,
                  max_ram_mb=500, iter_dat='oct_iters.dat', iter_stop=100,
                  tau_dat='oct_tau.dat', params_file='oct_params.dat',
                  limit_pulses=True, keep_pulses='all')
    model.user_data['initial_states'] = '10'
    model.user_data['target_states'] = '01'
    model.user_data['seed'] = 0
    return model


def generate_guess(rf='berlin_run/guess'):
    """Do a quantum trajectory simulation of the guess pulse"""
    if os.path.isfile(join(rf, 'qubit_pop.dat')):
        return
    mkdir(rf)
    oct_unidir_model().write_to_runfolder(rf)
    pc = qdyn_prop_traj(['--n-trajs=20', '--state-label=10', rf],
                        _out=join(rf, 'prop.log'))
    pc.wait()


def generate_rho_optimized(rf='berlin_run/rho_optimize'):
    """Optimize in Liouville space"""
    if os.path.isfile(join(rf, 'qubit_pop.dat')):
        return
    mkdir(rf)
    oct_unidir_model(set_observables=False, mcwf=False, lambda_a=1e-2)\
        .write_to_runfolder(rf)
    pc = qdyn_optimize(['--rho', '--debug', '--J_T=J_T_re', rf],
                       _out=join(rf, 'oct.log'))

    pc.wait()
    for pulse1 in sorted(glob(join(rf, 'pulse1.oct.dat.0*'))):
        pulse2 = pulse1.replace('pulse1', 'pulse2')
        ext = os.path.splitext(pulse1)[1]
        rf_prop = 'berlin_run/rho_optimize_prop' + ext
        oct_unidir_model().write_to_runfolder(rf_prop)
        shutil.copy(pulse1, join(rf_prop, 'pulse1.dat'))
        shutil.copy(pulse2, join(rf_prop, 'pulse2.dat'))
        if  os.path.isfile(join(rf_prop, 'qubit_pop.dat')):
            continue
        print("propagate %s" % rf_prop)
        pc = qdyn_prop_traj(['--n-trajs=20', '--state-label=10', rf_prop],
                            _out=join(rf, 'prop.log'))
        pc.wait()
        print("%.2e" % err_state_to_state(psi10, join(rf_prop, 'psi_final.dat.*')))


def generate_mcwf_optimized(rf='berlin_run/mcwf_optimize'):
    """Optimize in Liouville space"""
    if os.path.isfile(join(rf, 'qubit_pop.dat')):
        return
    mkdir(rf)
    oct_unidir_model(set_observables=False, mcwf=True, lambda_a=1e-2,
                     ) .write_to_runfolder(rf)
    pc = qdyn_optimize(['--n-trajs=20', '--J_T=J_T_sm', rf],
                       _out=join(rf, 'oct.log'))
    pc.wait()
    for pulse1 in sorted(glob(join(rf, 'pulse1.oct.dat.0*'))):
        pulse2 = pulse1.replace('pulse1', 'pulse2')
        ext = os.path.splitext(pulse1)[1]
        rf_prop = 'berlin_run/mcwf_optimize_prop' + ext
        oct_unidir_model().write_to_runfolder(rf_prop)
        shutil.copy(pulse1, join(rf_prop, 'pulse1.dat'))
        shutil.copy(pulse2, join(rf_prop, 'pulse2.dat'))
        if  os.path.isfile(join(rf_prop, 'qubit_pop.dat')):
            continue
        print("propagate %s" % rf_prop)
        pc = qdyn_prop_traj(['--n-trajs=20', '--state-label=10', rf_prop],
                            _out=join(rf, 'prop.log'))
        pc.wait()
        print("%.2e" % err_state_to_state(psi10, join(rf_prop, 'psi_final.dat.*')))


def main():
    generate_analytical()
    generate_guess()
    generate_rho_optimized()
    generate_mcwf_optimized()

if __name__ == "__main__":
    sys.exit(main())
