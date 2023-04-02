from __future__ import annotations

import cvxpy as cp
import mosek


def solve_mosek(prob: cp.Problem) -> None:
    try:
        mosek_params = {
            'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-7,  # Primal feasibility tolerance
            'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-7,  # Dual feasibility tolerance
            'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-7,  # Relative gap tolerance
            # 'MSK_IPAR_INTPNT_SOLVE_FORM': int(mosek.solveform.primal) # Solve primal problem formulation
        }
        # prob.solve(warm_start=True, solver=cp.MOSEK, mosek_params=mosek_params, verbose=True)
        prob.solve(warm_start=True, solver=cp.MOSEK)
        # prob.solve(warm_start=True, solver=cp.ECOS, verbose=True)
    except cp.SolverError:
        print('MOSEK solver failed. Trying cp.ECOS')
        prob.solve(solver=cp.ECOS)
    if prob.status != 'optimal':
        print(f'prob.status = {prob.status}')
        if 'infeasible' in prob.status:
            # your problem should never be infeasible. So now go debug
            import pdb
            pdb.set_trace()
