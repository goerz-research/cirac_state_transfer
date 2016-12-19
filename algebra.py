from collections import OrderedDict
from sympy import symbols

from qnet.algebra.abstract_algebra import extra_binary_rules
from qnet.algebra.operator_algebra import OperatorPlus, create_operator_pm_cc
from qnet.algebra.hilbert_space_algebra import LocalSpace, ProductSpace

def split_hamiltonian(H, use_cc=True):
    res = OrderedDict()
    H = H.expand().simplify_scalar()
    n_nodes = len(H.space.local_factors) // 2
    controls = [(symbols('Omega_%d' % (i+1))) for i in range(n_nodes)]
    Hdrift = H.substitute({control: 0 for control in controls})
    res['H0'] = OperatorPlus.create(
            *[H for H in Hdrift.operands if isinstance(H.space, LocalSpace)])
    res['Hint'] = OperatorPlus.create(
            *[H for H in Hdrift.operands if isinstance(H.space, ProductSpace)]
            ).expand().simplify_scalar()
    Hdrive = (H - Hdrift).expand()
    def all_zero_except(controls, i):
        return {control: 0
                for (j, control) in enumerate(controls)
                if j != i}
    for i in range(n_nodes):
        mapping = all_zero_except(controls, i)
        res['Hd_%d' % (i+1)] = Hdrive.substitute(mapping)
    if use_cc:
        for name, H in res.items():
            with extra_binary_rules(OperatorPlus, create_operator_pm_cc()):
                res[name] = H.simplify_scalar().simplify()
    return res
