from __future__ import annotations

import cvxpy as cp


def solve_mosek(prob: cp.Problem) -> None:
    try:
        prob.solve(warm_start=True, solver=cp.MOSEK)
    except cp.SolverError:
        print('MOSEK solver failed. Trying cp.ECOS')
        prob.solve(solver=cp.ECOS)
    if prob.status != 'optimal':
        print(f'prob.status = {prob.status}')
        if 'infeasible' in prob.status:
            # your problem should never be infeasible. So now go debug
            import pdb
            pdb.set_trace()
