"""Routines to construct the QDYN model for a two-node network"""

from os.path import join
from glob import glob
import shutil

import qutip
import QDYN

from two_node_slh import dagger
from qnet.convert.to_qutip import convert_to_qutip


def state(SYS, *numbers, fmt='qutip'):
    """Construct a state for a given QNET space by giving a quantum number for
    each sub-space"""
    states = []
    assert len(numbers) == len(SYS.H.space.local_factors)
    for i, hs in enumerate(SYS.H.space.local_factors):
        states.append(qutip.basis(hs.dimension, numbers[i]))
    if fmt == 'qutip':
        return qutip.tensor(*states)
    elif fmt == 'numpy':
        return QDYN.linalg.vectorize(qutip.tensor(*states).data.todense())
    else:
        raise ValueError("Unknown fmt")


def err_state_to_state(target_state, final_states_glob):
    """Return error of a single state-to-state transfer

    Args:
        target_state: numpy array of amplitudes of target state
        final_states_glob: string that expands to a list of files
            from which propagated states should be read (for multiple
            trajectories)
    """
    final_state_files = glob(final_states_glob)
    target_state = QDYN.linalg.vectorize(target_state)
    n = len(final_state_files)
    assert n > 0
    F = 0.0
    n_hilbert = len(target_state)
    for fn in final_state_files:
        final_state = QDYN.state.read_psi_amplitudes(fn, n=n_hilbert)
        F += abs(QDYN.linalg.inner(target_state, final_state))**2
    return 1.0 - F/float(n)


def make_qdyn_model(SYS, Delta, g, kappa, Sym1, Op1, Sym2, Op2, pulse1, pulse2,
        mcwf=False, non_herm=False, states=None, set_observables=True):
    """Construct a QDYN LevelModel for a two-node network, configured for
    propagation

    Args:
        SYS: SLH model for SYS.H and SYS.L
        Delta: Value of detuning of laser freq from atomic transition (MHz)
        g: Value of qubit-cavity coupling (MHz)
        kappa: (half) the leakage rate of each cavity (MHz)
        Sym1: dictionary of sympy symbols for system 1
        Op1: dictionary of QNET operators for system 1
        Sym2: dictionary of sympy symbols for system 2
        Op2: dictionary of QNET operators for system 2
        pulse1: a QDYN.pulse.Pulse instance driving system 1. The time grid
            parameters will be taken from this pulse
        pulse2: a QDYN.pulse.Pulse instance driving system 2
        mcwf: If True, initialize for MCWF propagation. Otherwise,
            non-dissipative Hilbert space or Liouville space propagation
        non_herm: If True, add the decay term to the Hamiltonian. If used
            together with `mcwf=True`, this improves efficiency slightly. If
            used with `mwwf=False`, a non-Hermitian Hamiltonian would be
            propagated
        states: dict label => state (numpy array of amplitudes)
        set_observables: If True, define observables in the model
    """
    nt = len(pulse1.tgrid) + 1
    t0 = pulse1.t0
    T = pulse1.T

    num_vals = {
        Sym1['Delta']: float(Delta),
        Sym1['g']:     float(g),
        Sym2['Delta']: float(Delta),
        Sym2['g']:     float(g),
        Sym1['I']: 1j,
        Sym2['I']: 1j,
        Sym1['kappa']: float(kappa),
        Sym2['kappa']: float(kappa),
    }

    construct_mcwf_ham = False
    if mcwf:
        construct_mcwf_ham = True
    if non_herm:
        construct_mcwf_ham = False

    H_num = SYS.H.substitute(num_vals)

    # drift Hamiltonian
    H0 = convert_to_qutip(H_num.substitute({Sym1['Omega']:0, Sym2['Omega']:0}))

    # control Hamiltonian (qubit 1)
    H1_1 = convert_to_qutip(
              H_num.substitute({Sym1['Omega']**2:0, Sym2['Omega']**2:0})
              .substitute({Sym1['Omega']:1, Sym2['Omega']:0})) - H0

    # control Hamiltonian (qubit 2)
    H1_2 = convert_to_qutip(
              H_num.substitute({Sym1['Omega']**2:0, Sym2['Omega']**2:0})
              .substitute({Sym1['Omega']:0, Sym2['Omega']:1})) - H0

    # Stark shift Hamiltonian (qubit 1)
    #H2_1 = H_num.substitute({Sym1['Omega']**2:1, Sym2['Omega']**2:0})\
    #          .substitute({Sym1['Omega']:0, Sym2['Omega']:0}) \
    #          .to_qutip() - H0

    # Stark shift Hamiltonian (qubit 2)
    #H2_2 = H_num.substitute({Sym1['Omega']**2:0, Sym2['Omega']**2:1})\
    #          .substitute({Sym1['Omega']:0, Sym2['Omega']:0}) \
    #          .to_qutip() - H0

    L = convert_to_qutip(SYS.L[0,0].substitute(num_vals))

    model = QDYN.model.LevelModel()

    if non_herm:
        H0 = H0 - 0.5j*(L.dag() * L)
    else:
        model.add_lindblad_op(L, op_unit='sqrt_MHz', add_to_H_jump='indexed')

    model.add_ham(H0, op_unit='MHz', op_type='potential')
    model.add_ham(H1_1, pulse=pulse1, op_unit='dimensionless',
                  op_type='dipole')
    model.add_ham(H1_2, pulse=pulse2, op_unit='dimensionless',
                  op_type='dipole')
    #model.add_ham(H2_1, op_unit='MHz^-1', op_type='dstark')
    #model.add_ham(H2_2, op_unit='MHz^-1', op_type='dstark')

    model.set_propagation(T=T, nt=nt, t0=t0, time_unit='microsec',
                          prop_method='newton', use_mcwf=mcwf, mcwf_order=2,
                          construct_mcwf_ham=construct_mcwf_ham)
    if states is not None:
        for label, psi in states.items():
            assert psi.shape[0] == H0.shape[0]
            model.add_state(psi, label)

    # Observables
    if set_observables:
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1,1)]:
            model.add_observable(state(SYS, i,j,0,0)
                                    * state(SYS, i,j,0,0).dag(),
                                 outfile='qubit_pop.dat',
                                 exp_unit='dimensionless',
                                 time_unit='microsec',
                                 col_label='P(%d%d)' % (i, j),
                                 is_real=True)
        model.add_observable(state(SYS, 0,0,0,1) * state(SYS, 0,0,0,1).dag(),
                        outfile='beta_pop.dat', exp_unit='dimensionless',
                        time_unit='microsec', col_label='P(0001)',
                        is_real=True)
        model.add_observable(state(SYS, 0,0,1,0) * state(SYS, 0,0,1,0).dag(),
                        outfile='beta_pop.dat', exp_unit='dimensionless',
                        time_unit='microsec', col_label='P(0010)',
                        is_real=True)
        Op_n1 = convert_to_qutip(dagger(Op1['a']) * Op1['a'])
        Op_n2 = convert_to_qutip(dagger(Op2['a']) * Op2['a'])
        model.add_observable(Op_n1,
                        outfile='cavity_excitation.dat',
                        exp_unit='dimensionless',
                        time_unit='microsec', col_label='<n_1>',
                        is_real=True)
        model.add_observable(Op_n2,
                        outfile='cavity_excitation.dat',
                        exp_unit='dimensionless',
                        time_unit='microsec', col_label='<n_2>',
                        is_real=True)
        model.add_observable(L.dag()*L,
                        outfile='cavity_excitation.dat',
                        exp_unit='dimensionless',
                        time_unit='microsec', col_label='<L^+L>',
                        is_real=True)
    model.user_data['time_unit'] = 'microsec'
    model.user_data['write_jump_record'] = 'jump_record.dat'
    model.user_data['write_final_state'] = 'psi_final.dat'
    model.user_data['seed'] = 0
    return model


def prepare_rf_prop(model, rf_oct, *rf_prop, oct_pulses='pulse*.oct.dat'):
    """Generate propagation runfolders by writing model to every propagation
    runfolder and copying the optimized pulses from rf_oct"""
    for rf in rf_prop:
        model.write_to_runfolder(rf)
        for file in glob(join(rf_oct, oct_pulses)):
            shutil.copy(file, rf)
            shutil.copy(file, rf)
