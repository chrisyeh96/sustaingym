from __future__ import annotations

import cvxpy as cp


def solve_mosek(prob: cp.Problem, verbose: int = 0) -> None:
    """Uses cvxpy solvers to solve optimization problem.

    Args:
        prob: optimization problem
        verbose: if >= 2, prints if MOSEK solver failed
    """
    try:
        prob.solve(warm_start=True, solver=cp.MOSEK)
    except cp.SolverError:
        prob.solve(solver=cp.ECOS)
        if verbose >= 2:
            print('Default MOSEK solver failed in action projection. Trying ECOS. ')
            if prob.status != 'optimal':
                print(f'prob.status = {prob.status}')
        if 'infeasible' in prob.status:
            # your problem should never be infeasible. So now go debug
            import pdb
            pdb.set_trace()  # :)
